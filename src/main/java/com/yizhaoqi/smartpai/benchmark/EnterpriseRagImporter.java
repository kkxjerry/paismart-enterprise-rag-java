package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.time.Duration;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/** Streams EnterpriseRAG JSONL documents into an isolated Elasticsearch index. */
public final class EnterpriseRagImporter {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int CHECKPOINT_VERSION = 1;

    private EnterpriseRagImporter() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        validateFiles(config);
        JsonHttpClient http = new JsonHttpClient();
        validateIndex(http, config);
        Map<String, AclDocument> aclByDocId = loadAcl(config.aclDocs());
        EmbeddingClient embeddings = new EmbeddingClient(config.embeddingConfig());
        ImportState state = loadCheckpoint(config);

        long startingDocuments = state.documentsProcessed();
        long startingChunks = state.chunksIndexed();
        long started = System.nanoTime();
        try {
            streamDocuments(config, state, aclByDocId, embeddings, http, started, startingDocuments);
        } catch (Exception exception) {
            appendFailure(config.failureLog(), state, exception);
            throw exception;
        }

        http.requireJson(
                "POST",
                baseUrl(config) + "/_refresh",
                null,
                "",
                config.maxRetries(),
                Duration.ofSeconds(60));
        state.setUpdatedAt(now());
        saveCheckpoint(config.checkpoint(), state);

        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("event", "import_complete");
        summary.put("index", config.index());
        summary.put("embedding_model", config.embeddingModel());
        summary.put("embedding_dimension", config.embeddingDimension());
        summary.put("documents_processed", state.documentsProcessed());
        summary.put("chunks_indexed", state.chunksIndexed());
        summary.put("batches_completed", state.batchesCompleted());
        summary.put("documents_processed_this_run", state.documentsProcessed() - startingDocuments);
        summary.put("chunks_indexed_this_run", state.chunksIndexed() - startingChunks);
        summary.put("elapsed_seconds_this_run", elapsedSeconds(started));
        summary.put("checkpoint", config.checkpoint().toAbsolutePath().normalize().toString());
        String json = MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(summary);
        System.out.println(json);
        if (config.output() != null) {
            createParent(config.output());
            Files.writeString(config.output(), json + System.lineSeparator(), StandardCharsets.UTF_8);
        }
    }

    private static void streamDocuments(
            Config config,
            ImportState state,
            Map<String, AclDocument> aclByDocId,
            EmbeddingClient embeddings,
            JsonHttpClient http,
            long started,
            long startingDocuments) throws Exception {
        if (config.maxDocuments() > 0 && state.documentsProcessed() >= config.maxDocuments()) {
            return;
        }
        List<JsonNode> batch = new ArrayList<>(config.documentBatchSize());
        int lineNumber = 0;
        int lastBatchLine = state.nextLine() - 1;
        try (BufferedReader reader = Files.newBufferedReader(config.docs(), StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (lineNumber < state.nextLine() || line.isBlank()) {
                    continue;
                }
                batch.add(parseJsonLine(config.docs(), lineNumber, line));
                lastBatchLine = lineNumber;
                boolean limitReached = config.maxDocuments() > 0
                        && state.documentsProcessed() + batch.size() >= config.maxDocuments();
                if (batch.size() >= config.documentBatchSize() || limitReached) {
                    completeBatch(config, state, batch, lastBatchLine, aclByDocId, embeddings, http);
                    printProgress(state, started, startingDocuments);
                    batch.clear();
                }
                if (limitReached) {
                    break;
                }
            }
        }
        if (!batch.isEmpty()) {
            completeBatch(config, state, batch, lastBatchLine, aclByDocId, embeddings, http);
            printProgress(state, started, startingDocuments);
        }
    }

    private static void completeBatch(
            Config config,
            ImportState state,
            List<JsonNode> documents,
            int lastLine,
            Map<String, AclDocument> aclByDocId,
            EmbeddingClient embeddings,
            JsonHttpClient http) throws Exception {
        List<Chunk> chunks = buildChunks(config, documents, aclByDocId);
        List<String> passages = chunks.stream()
                .map(chunk -> chunk.source().path("title").asText() + "\n" + chunk.source().path("textContent").asText())
                .toList();
        List<List<Double>> vectors = embedAll(config, embeddings, passages);
        if (vectors.size() != chunks.size()) {
            throw new IllegalStateException("embedding count does not match chunk count");
        }
        for (int index = 0; index < chunks.size(); index++) {
            chunks.get(index).source().set("vector", MAPPER.valueToTree(vectors.get(index)));
        }
        bulkIndex(config, http, chunks);

        state.setNextLine(lastLine + 1);
        state.setDocumentsProcessed(state.documentsProcessed() + documents.size());
        state.setChunksIndexed(state.chunksIndexed() + chunks.size());
        state.setBatchesCompleted(state.batchesCompleted() + 1);
        state.setUpdatedAt(now());
        saveCheckpoint(config.checkpoint(), state);
    }

    private static List<List<Double>> embedAll(
            Config config,
            EmbeddingClient embeddings,
            List<String> texts) throws Exception {
        List<List<String>> batches = new ArrayList<>();
        for (int start = 0; start < texts.size(); start += config.embeddingBatchSize()) {
            batches.add(List.copyOf(texts.subList(start, Math.min(start + config.embeddingBatchSize(), texts.size()))));
        }
        if (config.embeddingWorkers() == 1 || batches.size() <= 1) {
            List<List<Double>> vectors = new ArrayList<>(texts.size());
            for (List<String> batch : batches) {
                vectors.addAll(embeddings.embed(batch, EmbeddingClient.InputType.DOCUMENT));
            }
            return vectors;
        }

        ExecutorService executor = Executors.newFixedThreadPool(config.embeddingWorkers());
        try {
            List<Future<List<List<Double>>>> futures = batches.stream()
                    .map(batch -> executor.submit(() -> embeddings.embed(batch, EmbeddingClient.InputType.DOCUMENT)))
                    .toList();
            List<List<Double>> vectors = new ArrayList<>(texts.size());
            for (Future<List<List<Double>>> future : futures) {
                try {
                    vectors.addAll(future.get());
                } catch (ExecutionException exception) {
                    Throwable cause = exception.getCause();
                    if (cause instanceof Exception nested) {
                        throw nested;
                    }
                    throw exception;
                }
            }
            return vectors;
        } finally {
            executor.shutdownNow();
        }
    }

    static List<Chunk> buildChunks(
            Config config,
            List<JsonNode> documents,
            Map<String, AclDocument> aclByDocId) {
        List<Chunk> chunks = new ArrayList<>();
        String indexedAt = now();
        for (JsonNode document : documents) {
            String docId = requiredText(document, "doc_id");
            String title = document.path("title").asText("");
            String sourceType = document.path("source_type").asText("unknown");
            AclDocument acl = aclByDocId.getOrDefault(docId, AclDocument.empty());
            List<String> parts = TextChunker.chunk(
                    document.path("text").asText(""),
                    config.chunkSize(),
                    config.chunkOverlap());
            for (int index = 0; index < parts.size(); index++) {
                String text = parts.get(index);
                int chunkId = index + 1;
                ObjectNode source = MAPPER.createObjectNode();
                source.put("benchmarkDocId", docId);
                source.put("chunkId", chunkId);
                source.put("title", title);
                source.put("textContent", text);
                source.put("sourceType", sourceType);
                source.put("sourcePath", document.path("source_path").asText(""));
                source.put("sourceDataset", document.path("source_dataset").asText(""));
                source.put("tenantId", acl.tenantId());
                source.put("classification", acl.classification());
                source.set("allowedGroupIds", MAPPER.valueToTree(acl.allowedGroupIds()));
                source.set("deniedGroupIds", MAPPER.valueToTree(acl.deniedGroupIds()));
                source.put("modelVersion", config.embeddingModel());
                source.put("contentHash", sha256(text));
                source.put("indexedAt", indexedAt);
                chunks.add(new Chunk(docId + ":" + String.format("%05d", chunkId), source));
            }
        }
        return chunks;
    }

    private static void bulkIndex(Config config, JsonHttpClient http, List<Chunk> chunks) throws Exception {
        for (int start = 0; start < chunks.size(); start += config.bulkSize()) {
            List<Chunk> batch = chunks.subList(start, Math.min(start + config.bulkSize(), chunks.size()));
            StringBuilder ndjson = new StringBuilder();
            for (Chunk chunk : batch) {
                ObjectNode action = MAPPER.createObjectNode();
                ObjectNode index = action.putObject("index");
                index.put("_index", config.index());
                index.put("_id", chunk.id());
                ndjson.append(MAPPER.writeValueAsString(action)).append('\n');
                ndjson.append(MAPPER.writeValueAsString(chunk.source())).append('\n');
            }

            JsonNode response = null;
            for (int attempt = 0; attempt <= config.maxRetries(); attempt++) {
                response = http.requireNdjson(
                        stripTrailingSlash(config.esUrl()) + "/_bulk",
                        ndjson.toString().getBytes(StandardCharsets.UTF_8),
                        config.maxRetries(),
                        Duration.ofMinutes(2));
                if (!response.path("errors").asBoolean(false)) {
                    break;
                }
                if (!allBulkFailuresRetryable(response) || attempt >= config.maxRetries()) {
                    throw new IllegalStateException("bulk indexing failed: " + firstBulkFailures(response));
                }
                Thread.sleep(1_000L << Math.min(attempt, 5));
            }
        }
    }

    private static boolean allBulkFailuresRetryable(JsonNode response) {
        boolean found = false;
        for (JsonNode item : response.path("items")) {
            JsonNode result = item.path("index");
            if (result.has("error")) {
                found = true;
                int status = result.path("status").asInt();
                if (status != 429 && status < 500) {
                    return false;
                }
            }
        }
        return found;
    }

    private static String firstBulkFailures(JsonNode response) throws IOException {
        ArrayNode failures = MAPPER.createArrayNode();
        for (JsonNode item : response.path("items")) {
            JsonNode result = item.path("index");
            if (result.has("error") && failures.size() < 3) {
                failures.add(result);
            }
        }
        return MAPPER.writeValueAsString(failures);
    }

    private static Map<String, AclDocument> loadAcl(Path path) throws IOException {
        Map<String, AclDocument> values = new LinkedHashMap<>();
        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            int lineNumber = 0;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.isBlank()) {
                    continue;
                }
                JsonNode row = parseJsonLine(path, lineNumber, line);
                String docId = requiredText(row, "doc_id");
                values.put(docId, new AclDocument(
                        row.path("tenant_id").asText(""),
                        row.path("classification").asText(""),
                        strings(row.path("allowed_group_ids")),
                        strings(row.path("denied_group_ids"))));
            }
        }
        return Map.copyOf(values);
    }

    private static void validateIndex(JsonHttpClient http, Config config) throws Exception {
        JsonNode mapping = http.requireJson(
                "GET", baseUrl(config) + "/_mapping", null, "", 1, Duration.ofSeconds(30));
        JsonNode indexMapping = mapping.path(config.index());
        if (indexMapping.isMissingNode() && mapping.fields().hasNext()) {
            indexMapping = mapping.fields().next().getValue();
        }
        JsonNode properties = indexMapping.path("mappings").path("properties");
        int dims = properties.path("vector").path("dims").asInt(-1);
        if (dims != config.embeddingDimension()) {
            throw new IllegalStateException(
                    "index vector dimension mismatch: expected " + config.embeddingDimension() + ", got " + dims);
        }
        if (!"standard".equals(properties.path("textContent").path("analyzer").asText())) {
            throw new IllegalStateException("EnterpriseRAG textContent must use the standard analyzer");
        }
    }

    private static ImportState loadCheckpoint(Config config) throws IOException {
        Map<String, Object> signature = signature(config);
        if (!Files.exists(config.checkpoint())) {
            return ImportState.create(signature);
        }
        JsonNode row = MAPPER.readTree(config.checkpoint().toFile());
        if (row.path("version").asInt() != CHECKPOINT_VERSION) {
            throw new IllegalArgumentException("unsupported checkpoint version");
        }
        if (!signatureMatches(row.path("signature"), signature)) {
            throw new IllegalArgumentException(
                    "checkpoint does not match docs, index, embedding model, or chunk settings");
        }
        return MAPPER.treeToValue(row, ImportState.class);
    }

    static boolean signatureMatches(JsonNode actual, Map<String, Object> expected) {
        if (!actual.isObject() || actual.size() != expected.size()) {
            return false;
        }
        for (Map.Entry<String, Object> entry : expected.entrySet()) {
            JsonNode expectedValue = MAPPER.valueToTree(entry.getValue());
            JsonNode actualValue = actual.path(entry.getKey());
            if (actualValue.isMissingNode() || !actualValue.asText().equals(expectedValue.asText())) {
                return false;
            }
        }
        return true;
    }

    private static Map<String, Object> signature(Config config) throws IOException {
        Map<String, Object> signature = new LinkedHashMap<>();
        Path docs = config.docs().toAbsolutePath().normalize();
        signature.put("docs_path", docs.toString());
        signature.put("docs_size", Files.size(docs));
        signature.put("docs_mtime_ms", Files.getLastModifiedTime(docs).toMillis());
        signature.put("index", config.index());
        signature.put("embedding_model", config.embeddingModel());
        signature.put("embedding_dimension", config.embeddingDimension());
        signature.put("chunk_size", config.chunkSize());
        signature.put("chunk_overlap", config.chunkOverlap());
        return signature;
    }

    private static void saveCheckpoint(Path path, ImportState state) throws IOException {
        createParent(path);
        Path temporary = path.resolveSibling(path.getFileName() + ".tmp");
        MAPPER.writerWithDefaultPrettyPrinter().writeValue(temporary.toFile(), state);
        try {
            Files.move(temporary, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException ignored) {
            Files.move(temporary, path, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static void appendFailure(Path path, ImportState state, Exception exception) {
        try {
            createParent(path);
            ObjectNode row = MAPPER.createObjectNode();
            row.put("failed_at", now());
            row.put("next_line", state.nextLine());
            row.put("documents_processed", state.documentsProcessed());
            row.put("error_type", exception.getClass().getName());
            row.put("error", exception.getMessage());
            try (BufferedWriter writer = Files.newBufferedWriter(
                    path,
                    StandardCharsets.UTF_8,
                    java.nio.file.StandardOpenOption.CREATE,
                    java.nio.file.StandardOpenOption.APPEND)) {
                writer.write(MAPPER.writeValueAsString(row));
                writer.newLine();
            }
        } catch (IOException logFailure) {
            exception.addSuppressed(logFailure);
        }
    }

    private static JsonNode parseJsonLine(Path path, int lineNumber, String line) throws IOException {
        try {
            return MAPPER.readTree(line);
        } catch (IOException exception) {
            throw new IOException("invalid JSONL at " + path + ":" + lineNumber, exception);
        }
    }

    private static List<String> strings(JsonNode node) {
        List<String> values = new ArrayList<>();
        if (node.isTextual()) {
            values.add(node.asText());
        } else if (node.isArray()) {
            node.forEach(value -> values.add(value.asText()));
        }
        return List.copyOf(values);
    }

    private static String requiredText(JsonNode node, String field) {
        String value = node.path(field).asText();
        if (value.isBlank()) {
            throw new IllegalArgumentException("document is missing " + field);
        }
        return value;
    }

    private static String sha256(String text) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(digest.digest(text.getBytes(StandardCharsets.UTF_8)));
        } catch (java.security.NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is unavailable", exception);
        }
    }

    private static void printProgress(ImportState state, long started, long startingDocuments) throws IOException {
        double elapsed = Math.max(elapsedSeconds(started), 0.001d);
        ObjectNode progress = MAPPER.createObjectNode();
        progress.put("event", "batch_complete");
        progress.put("documents_processed", state.documentsProcessed());
        progress.put("chunks_indexed", state.chunksIndexed());
        progress.put("next_line", state.nextLine());
        progress.put("documents_per_second_this_run", (state.documentsProcessed() - startingDocuments) / elapsed);
        System.out.println(MAPPER.writeValueAsString(progress));
    }

    private static void validateFiles(Config config) throws IOException {
        if (!Files.isRegularFile(config.docs()) || !Files.isRegularFile(config.aclDocs())) {
            throw new IllegalArgumentException("--docs and --acl-docs must be readable files");
        }
        if (config.chunkOverlap() >= config.chunkSize()) {
            throw new IllegalArgumentException("--chunk-overlap must be smaller than --chunk-size");
        }
        createParent(config.checkpoint());
        createParent(config.failureLog());
    }

    private static void createParent(Path path) throws IOException {
        Path parent = path.toAbsolutePath().normalize().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
    }

    private static String baseUrl(Config config) {
        return stripTrailingSlash(config.esUrl()) + "/" + config.index();
    }

    private static String stripTrailingSlash(String value) {
        return value.endsWith("/") ? value.substring(0, value.length() - 1) : value;
    }

    private static String now() {
        return OffsetDateTime.now(ZoneOffset.UTC).toString();
    }

    private static double elapsedSeconds(long started) {
        return (System.nanoTime() - started) / 1_000_000_000.0d;
    }

    record Config(
            Path docs,
            Path aclDocs,
            String esUrl,
            String index,
            String embeddingUrl,
            String embeddingModel,
            int embeddingDimension,
            String embeddingApiFormat,
            String embeddingApiKey,
            String embeddingQueryInstruction,
            int embeddingBatchSize,
            int embeddingWorkers,
            int documentBatchSize,
            int bulkSize,
            int chunkSize,
            int chunkOverlap,
            int maxRetries,
            long maxDocuments,
            Path checkpoint,
            Path failureLog,
            Path output) {

        static Config parse(String[] args) {
            Arguments values = Arguments.parse(args);
            String apiFormat = values.oneOf(
                    "embedding-api-format", "local", Set.of("local", "openai", "dashscope"));
            String keyEnvironment = values.string("embedding-api-key-env", "DASHSCOPE_API_KEY");
            String apiKey = System.getenv().getOrDefault(keyEnvironment, "");
            String outputValue = values.string("output", "");
            long maxDocuments = Long.parseLong(values.string("max-documents", "0"));
            if (maxDocuments < 0) {
                throw new IllegalArgumentException("--max-documents must not be negative");
            }
            return new Config(
                    values.requiredPath("docs"),
                    values.requiredPath("acl-docs"),
                    values.string("es-url", "http://127.0.0.1:19200"),
                    values.required("index"),
                    values.string("embedding-url", "http://127.0.0.1:18084/v1/embeddings"),
                    values.string("embedding-model", "Qwen/Qwen3-Embedding-4B"),
                    values.positiveInt("embedding-dimension", 2048),
                    apiFormat,
                    apiKey,
                    values.string("embedding-query-instruction", ""),
                    values.positiveInt("embedding-batch-size", 32),
                    values.positiveInt("embedding-workers", 1),
                    values.positiveInt("document-batch-size", 20),
                    values.positiveInt("bulk-size", 100),
                    values.positiveInt("chunk-size", 1200),
                    values.nonNegativeInt("chunk-overlap", 200),
                    values.nonNegativeInt("max-retries", 3),
                    maxDocuments,
                    values.path("checkpoint", "runs/import-checkpoint.json"),
                    values.path("failure-log", "runs/import-failures.jsonl"),
                    outputValue.isBlank() ? null : Path.of(outputValue));
        }

        EmbeddingClient.Config embeddingConfig() {
            return new EmbeddingClient.Config(
                    embeddingUrl,
                    embeddingModel,
                    embeddingDimension,
                    embeddingApiFormat,
                    embeddingApiKey,
                    embeddingQueryInstruction,
                    maxRetries,
                    Duration.ofMinutes(2));
        }
    }

    record Chunk(String id, ObjectNode source) {
    }

    record AclDocument(
            String tenantId,
            String classification,
            List<String> allowedGroupIds,
            List<String> deniedGroupIds) {

        static AclDocument empty() {
            return new AclDocument("", "", List.of(), List.of());
        }
    }

    public static final class ImportState {
        public int version;
        public Map<String, Object> signature;
        public int nextLine;
        public long documentsProcessed;
        public long chunksIndexed;
        public long batchesCompleted;
        public String startedAt;
        public String updatedAt;

        public ImportState() {
        }

        static ImportState create(Map<String, Object> signature) {
            ImportState state = new ImportState();
            state.version = CHECKPOINT_VERSION;
            state.signature = Map.copyOf(signature);
            state.nextLine = 1;
            state.startedAt = now();
            state.updatedAt = state.startedAt;
            return state;
        }

        public int version() {
            return version;
        }

        public Map<String, Object> signature() {
            return signature;
        }

        public int nextLine() {
            return nextLine;
        }

        public long documentsProcessed() {
            return documentsProcessed;
        }

        public long chunksIndexed() {
            return chunksIndexed;
        }

        public long batchesCompleted() {
            return batchesCompleted;
        }

        public String startedAt() {
            return startedAt;
        }

        public String updatedAt() {
            return updatedAt;
        }

        void setNextLine(int value) {
            nextLine = value;
        }

        void setDocumentsProcessed(long value) {
            documentsProcessed = value;
        }

        void setChunksIndexed(long value) {
            chunksIndexed = value;
        }

        void setBatchesCompleted(long value) {
            batchesCompleted = value;
        }

        void setUpdatedAt(String value) {
            updatedAt = value;
        }
    }
}
