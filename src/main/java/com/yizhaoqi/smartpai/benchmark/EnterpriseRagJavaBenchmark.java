package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.yizhaoqi.smartpai.service.Bm25QueryRewriter;
import com.yizhaoqi.smartpai.service.ReciprocalRankFusion;

import java.io.BufferedWriter;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Runs the production-shaped Java Dense + BM25 + RRF retrieval path against
 * EnterpriseRAG without starting Spring or connecting to application databases.
 */
public final class EnterpriseRagJavaBenchmark {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Pattern TOKEN_PATTERN = Pattern.compile(
            "[a-z0-9_][a-z0-9_./:+#-]*|[\\u4e00-\\u9fff]",
            Pattern.CASE_INSENSITIVE);

    private EnterpriseRagJavaBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        if (Set.of("openai", "dashscope").contains(config.embeddingApiFormat())
                && config.embeddingApiKey().isBlank()) {
            throw new IllegalArgumentException(
                    "DASHSCOPE_API_KEY or --embedding-api-key is required for cloud embeddings");
        }
        List<Question> questions = loadQuestions(config.questions());
        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();

        Files.createDirectories(config.output().toAbsolutePath().getParent());
        Files.createDirectories(config.detailsOutput().toAbsolutePath().getParent());

        List<Result> results = new ArrayList<>(questions.size());
        long runStarted = System.nanoTime();
        try (BufferedWriter details = Files.newBufferedWriter(config.detailsOutput(), StandardCharsets.UTF_8)) {
            for (int index = 0; index < questions.size(); index++) {
                Question question = questions.get(index);
                Result result = retrieve(client, config, question);
                results.add(result);
                details.write(MAPPER.writeValueAsString(result.toDetails(question)));
                details.newLine();

                int completed = index + 1;
                if (config.progressEvery() > 0
                        && (completed % config.progressEvery() == 0 || completed == questions.size())) {
                    double seconds = (System.nanoTime() - runStarted) / 1_000_000_000.0d;
                    System.out.printf(
                            "{\"event\":\"java_retrieval_progress\",\"completed\":%d,\"total\":%d,\"qps\":%.2f}%n",
                            completed,
                            questions.size(),
                            completed / Math.max(seconds, 0.001d));
                }
            }
        }

        Map<String, Object> summary = summarize(config, questions, results);
        MAPPER.writerWithDefaultPrettyPrinter().writeValue(config.output().toFile(), summary);
        System.out.println(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(summary));
    }

    static Result retrieve(HttpClient client, Config config, Question question) throws Exception {
        long started = System.nanoTime();
        ObjectNode filter = sourceAclFilter(question.sourceTypes());
        double embeddingMs = 0.0d;
        double denseMs = 0.0d;
        List<RankedDocument> dense = List.of();
        if (!"bm25".equals(config.retrievalMode())) {
            long embeddingStarted = System.nanoTime();
            List<Double> vector = embed(client, config, question.text());
            embeddingMs = elapsedMs(embeddingStarted);
            long denseStarted = System.nanoTime();
            dense = denseSearch(client, config, vector, filter);
            denseMs = elapsedMs(denseStarted);
        }

        double bm25Ms = 0.0d;
        List<RankedDocument> bm25 = List.of();
        double keywordBm25Ms = 0.0d;
        List<RankedDocument> keywordBm25 = List.of();
        double englishBm25Ms = 0.0d;
        List<RankedDocument> englishBm25 = List.of();
        if (!"dense".equals(config.retrievalMode())) {
            long bm25Started = System.nanoTime();
            bm25 = bm25Search(client, config, question.text(), filter, standardBm25Fields());
            bm25Ms = elapsedMs(bm25Started);

            String keywordQuery = Bm25QueryRewriter.keywordQuery(question.text());
            if (config.keywordBm25Enabled()
                    && Bm25QueryRewriter.isDistinctVariant(question.text(), keywordQuery)) {
                long keywordStarted = System.nanoTime();
                keywordBm25 = bm25Search(client, config, keywordQuery, filter, standardBm25Fields());
                keywordBm25Ms = elapsedMs(keywordStarted);
            }
            if (config.englishBm25Enabled()) {
                long englishStarted = System.nanoTime();
                englishBm25 = bm25Search(client, config, question.text(), filter, englishBm25Fields());
                englishBm25Ms = elapsedMs(englishStarted);
            }
        }

        long fusionStarted = System.nanoTime();
        List<RankedDocument> fused;
        List<List<RankedDocument>> rankings = new ArrayList<>();
        List<Double> weights = new ArrayList<>();
        if (!"bm25".equals(config.retrievalMode())) {
            rankings.add(dense);
            weights.add(config.denseWeight());
        }
        if (!"dense".equals(config.retrievalMode())) {
            rankings.add(bm25);
            weights.add(config.bm25Weight());
            if (!keywordBm25.isEmpty()) {
                rankings.add(keywordBm25);
                weights.add(config.keywordBm25Weight());
            }
            if (!englishBm25.isEmpty()) {
                rankings.add(englishBm25);
                weights.add(config.englishBm25Weight());
            }
        }
        if (rankings.size() == 1) {
            fused = rankings.get(0).stream().limit(config.topK()).toList();
        } else {
            fused = ReciprocalRankFusion.fuse(
                            rankings,
                            weights,
                            RankedDocument::docId,
                            config.rrfK(),
                            config.topK())
                    .stream()
                    .map(item -> item.value().withScore(item.score()))
                    .toList();
        }
        double fusionMs = elapsedMs(fusionStarted);
        return new Result(
                fused,
                embeddingMs,
                denseMs,
                bm25Ms,
                keywordBm25Ms,
                englishBm25Ms,
                fusionMs,
                elapsedMs(started));
    }

    private static List<Double> embed(HttpClient client, Config config, String query) throws Exception {
        JsonNode response = post(
                client,
                config.embeddingUrl(),
                embeddingRequestBody(config, query),
                config.embeddingApiKey());
        JsonNode vector = "dashscope".equals(config.embeddingApiFormat())
                ? response.path("output").path("embeddings").path(0).path("embedding")
                : response.path("data").path(0).path("embedding");
        if (!vector.isArray() || vector.size() != config.embeddingDimension()) {
            throw new IllegalStateException(
                    "embedding response dimension mismatch: expected "
                            + config.embeddingDimension() + ", got " + vector.size());
        }
        return MAPPER.convertValue(vector, new TypeReference<>() { });
    }

    static ObjectNode embeddingRequestBody(Config config, String query) {
        ObjectNode body = MAPPER.createObjectNode();
        body.put("model", config.embeddingModel());
        if ("dashscope".equals(config.embeddingApiFormat())) {
            body.putObject("input").putArray("texts").add(query);
            ObjectNode parameters = body.putObject("parameters");
            parameters.put("text_type", "query");
            parameters.put("dimension", config.embeddingDimension());
            parameters.put("output_type", "dense");
            if (!config.embeddingQueryInstruction().isBlank()) {
                parameters.put("instruct", config.embeddingQueryInstruction());
            }
        } else {
            String embeddingQuery = config.embeddingQueryInstruction().isBlank()
                    ? query
                    : "Instruct: " + config.embeddingQueryInstruction() + "\nQuery:" + query;
            body.putArray("input").add(embeddingQuery);
        }
        if ("openai".equals(config.embeddingApiFormat())) {
            body.put("dimensions", config.embeddingDimension());
        } else if ("local".equals(config.embeddingApiFormat())) {
            body.put("input_type", "query");
            body.put("dimension", config.embeddingDimension());
        }
        if (!"dashscope".equals(config.embeddingApiFormat())) {
            body.put("encoding_format", "float");
        }
        return body;
    }

    private static List<RankedDocument> denseSearch(
            HttpClient client,
            Config config,
            List<Double> vector,
            ObjectNode filter) throws Exception {
        return parseHits(post(client, searchUrl(config), denseSearchBody(config, vector, filter)));
    }

    static ObjectNode denseSearchBody(Config config, List<Double> vector, ObjectNode filter) {
        ObjectNode body = commonSearchBody(config);
        if ("opensearch".equals(config.engine())) {
            ObjectNode vectorQuery = MAPPER.createObjectNode();
            vectorQuery.set("vector", MAPPER.valueToTree(vector));
            vectorQuery.put("k", config.denseChunkCandidates());
            vectorQuery.putObject("method_parameters").put("ef_search", config.denseNumCandidates());
            if (filter != null) {
                vectorQuery.set("filter", filter);
            }
            ObjectNode knn = MAPPER.createObjectNode();
            knn.set("vector", vectorQuery);
            body.set("query", MAPPER.createObjectNode().set("knn", knn));
        } else {
            ObjectNode knn = body.putObject("knn");
            knn.put("field", "vector");
            knn.set("query_vector", MAPPER.valueToTree(vector));
            knn.put("k", config.denseChunkCandidates());
            knn.put("num_candidates", config.denseNumCandidates());
            if (filter != null) {
                knn.set("filter", filter);
            }
        }
        return body;
    }

    private static List<RankedDocument> bm25Search(
            HttpClient client,
            Config config,
            String query,
            ObjectNode filter,
            List<String> fields) throws Exception {
        return parseHits(post(client, searchUrl(config), bm25SearchBody(config, query, filter, fields)));
    }

    static ObjectNode bm25SearchBody(
            Config config,
            String query,
            ObjectNode filter,
            List<String> fields) {
        ObjectNode body = commonSearchBody(config);
        ObjectNode multiMatch = MAPPER.createObjectNode();
        multiMatch.put("query", query);
        ArrayNode fieldArray = multiMatch.putArray("fields");
        fields.forEach(fieldArray::add);
        multiMatch.put("type", "best_fields");
        multiMatch.put("operator", "or");

        if (filter == null) {
            body.set("query", MAPPER.createObjectNode().set("multi_match", multiMatch));
        } else {
            ObjectNode bool = MAPPER.createObjectNode();
            bool.putArray("must").add(MAPPER.createObjectNode().set("multi_match", multiMatch));
            bool.putArray("filter").add(filter);
            body.set("query", MAPPER.createObjectNode().set("bool", bool));
        }
        return body;
    }

    private static List<String> standardBm25Fields() {
        return List.of("title^2.0", "textContent^1.0");
    }

    private static List<String> englishBm25Fields() {
        return List.of("title.english^2.0", "textContent.english^1.0");
    }

    private static ObjectNode commonSearchBody(Config config) {
        ObjectNode body = MAPPER.createObjectNode();
        body.put("size", config.retrieverK());
        body.put("track_total_hits", false);
        body.putObject("collapse").put("field", "benchmarkDocId");
        body.putArray("_source")
                .add("benchmarkDocId")
                .add("chunkId")
                .add("title")
                .add("textContent")
                .add("sourceType");
        return body;
    }

    static ObjectNode sourceAclFilter(List<String> sourceTypes) {
        if (sourceTypes.isEmpty()) {
            return null;
        }
        ArrayNode filters = MAPPER.createArrayNode();
        filters.add(termQuery("tenantId", "tenant_redwood"));
        filters.add(termsQuery("sourceType", sourceTypes));
        filters.add(termsQuery(
                "allowedGroupIds",
                sourceTypes.stream().map(value -> "source:" + value).toList()));

        ObjectNode bool = MAPPER.createObjectNode();
        bool.set("filter", filters);
        bool.putArray("must_not").add(termsQuery(
                "deniedGroupIds",
                sourceTypes.stream().map(value -> "source:" + value).toList()));
        return MAPPER.createObjectNode().set("bool", bool);
    }

    private static ObjectNode termQuery(String field, String value) {
        return MAPPER.createObjectNode().set(
                "term",
                MAPPER.createObjectNode().put(field, value));
    }

    private static ObjectNode termsQuery(String field, List<String> values) {
        return MAPPER.createObjectNode().set(
                "terms",
                MAPPER.createObjectNode().set(field, MAPPER.valueToTree(values)));
    }

    private static String searchUrl(Config config) {
        return config.esUrl() + "/" + config.index() + "/_search";
    }

    private static JsonNode post(HttpClient client, String url, JsonNode body) throws Exception {
        return post(client, url, body, "");
    }

    private static JsonNode post(
            HttpClient client,
            String url,
            JsonNode body,
            String bearerToken) throws Exception {
        byte[] requestBody = MAPPER.writeValueAsBytes(body);
        int maxAttempts = bearerToken.isBlank() ? 1 : 4;
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            HttpURLConnection connection = (HttpURLConnection) URI.create(url).toURL().openConnection();
            connection.setConnectTimeout(10_000);
            connection.setReadTimeout(60_000);
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json");
            if (!bearerToken.isBlank()) {
                connection.setRequestProperty("Authorization", "Bearer " + bearerToken);
            }
            connection.setFixedLengthStreamingMode(requestBody.length);
            connection.setDoOutput(true);
            try {
                connection.getOutputStream().write(requestBody);
                int status = connection.getResponseCode();
                var responseStream = status >= 200 && status < 300
                        ? connection.getInputStream()
                        : connection.getErrorStream();
                byte[] responseBody = responseStream == null ? new byte[0] : responseStream.readAllBytes();
                String response = new String(responseBody, StandardCharsets.UTF_8);
                if (status >= 200 && status < 300) {
                    return MAPPER.readTree(response);
                }
                if ((status == 429 || status >= 500) && attempt < maxAttempts) {
                    Thread.sleep(1_000L << (attempt - 1));
                    continue;
                }
                throw new IllegalStateException("HTTP " + status + " from " + url + ": " + response);
            } catch (IOException exception) {
                if (attempt >= maxAttempts) {
                    throw exception;
                }
                Thread.sleep(1_000L << (attempt - 1));
            } finally {
                connection.disconnect();
            }
        }
        throw new IllegalStateException("embedding request retry loop exhausted");
    }

    private static List<RankedDocument> parseHits(JsonNode response) {
        List<RankedDocument> documents = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        for (JsonNode hit : response.path("hits").path("hits")) {
            JsonNode source = hit.path("_source");
            String docId = source.path("benchmarkDocId").asText();
            if (docId.isBlank() || !seen.add(docId)) {
                continue;
            }
            documents.add(new RankedDocument(
                    docId,
                    hit.path("_id").asText(),
                    source.path("chunkId").asInt(),
                    source.path("sourceType").asText(),
                    source.path("title").asText(),
                    source.path("textContent").asText(),
                    hit.path("_score").asDouble()));
        }
        return documents;
    }

    private static List<Question> loadQuestions(Path path) throws Exception {
        JsonNode payload = MAPPER.readTree(path.toFile());
        JsonNode rows = payload.isArray() ? payload : payload.path("questions");
        List<Question> questions = new ArrayList<>();
        for (JsonNode row : rows) {
            questions.add(new Question(
                    row.path("id").asText(),
                    row.path("question").asText(),
                    strings(row.path("expected_doc_ids")),
                    strings(row.path("source_types")),
                    row.path("question_type").asText("unknown")));
        }
        return questions;
    }

    private static List<String> strings(JsonNode node) {
        List<String> values = new ArrayList<>();
        if (node.isTextual()) {
            values.add(node.asText());
        } else if (node.isArray()) {
            node.forEach(value -> values.add(value.asText()));
        }
        return values;
    }

    static Map<String, Object> summarize(Config config, List<Question> questions, List<Result> results) {
        if (questions.size() != results.size()) {
            throw new IllegalArgumentException("questions and results must have the same size");
        }
        List<Integer> allIndexes = new ArrayList<>(questions.size());
        for (int index = 0; index < questions.size(); index++) {
            allIndexes.add(index);
        }

        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("run_id", "paismart_java_enterpriserag_source_acl_" + config.retrievalMode());
        summary.put("created_at", OffsetDateTime.now().toString());
        summary.put("runtime", "java");
        summary.put("engine", config.engine());
        summary.put("index", config.index());
        summary.put("embedding_model", config.embeddingModel());
        summary.put("embedding_dimension", config.embeddingDimension());
        summary.put("embedding_query_instruction", config.embeddingQueryInstruction());
        summary.put("embedding_api_format", config.embeddingApiFormat());
        summary.put("retrieval", "source_acl_" + config.retrievalMode());
        summary.put("retriever_k", config.retrieverK());
        summary.put("rrf_k", config.rrfK());
        summary.put("dense_weight", config.denseWeight());
        summary.put("bm25_weight", config.bm25Weight());
        summary.put("keyword_bm25_enabled", config.keywordBm25Enabled());
        summary.put("keyword_bm25_weight", config.keywordBm25Weight());
        summary.put("english_bm25_enabled", config.englishBm25Enabled());
        summary.put("english_bm25_weight", config.englishBm25Weight());
        summary.put("top_k_documents", config.topK());
        summary.putAll(evaluateGroup(questions, results, allIndexes));
        summary.put("by_question_type", groupedMetrics(questions, results, false));
        summary.put("by_source_type", groupedMetrics(questions, results, true));
        return summary;
    }

    private static Map<String, Object> evaluateGroup(
            List<Question> questions,
            List<Result> results,
            List<Integer> indexes) {
        int evaluable = 0;
        Map<Integer, Integer> hits = new LinkedHashMap<>();
        for (int k : List.of(1, 5, 10, 20)) {
            hits.put(k, 0);
        }
        double reciprocalSum = 0.0d;
        int allGoldHit10 = 0;
        int noGoldDocCount = 0;
        int sourceFilterViolationCount = 0;
        List<Double> latencies = new ArrayList<>();
        List<Integer> contextTokens = new ArrayList<>();

        for (int index : indexes) {
            Question question = questions.get(index);
            Result result = results.get(index);
            latencies.add(result.latencyMs());
            contextTokens.add(result.documents().stream()
                    .limit(10)
                    .mapToInt(document -> tokenCount(document.title() + "\n" + document.text()))
                    .sum());
            if (!question.sourceTypes().isEmpty()
                    && result.documents().stream().anyMatch(
                            document -> !question.sourceTypes().contains(document.sourceType()))) {
                sourceFilterViolationCount++;
            }
            if (question.expectedDocIds().isEmpty()) {
                noGoldDocCount++;
                continue;
            }
            evaluable++;
            List<String> ranking = result.documents().stream().map(RankedDocument::docId).toList();
            Set<String> expected = new HashSet<>(question.expectedDocIds());
            int firstRank = firstRank(ranking, expected);
            for (int k : hits.keySet()) {
                if (firstRank > 0 && firstRank <= k) {
                    hits.put(k, hits.get(k) + 1);
                }
            }
            if (firstRank > 0 && firstRank <= 10) {
                reciprocalSum += 1.0d / firstRank;
            }
            if (new HashSet<>(ranking.subList(0, Math.min(10, ranking.size()))).containsAll(expected)) {
                allGoldHit10++;
            }
        }

        latencies.sort(Comparator.naturalOrder());
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("questions_total", indexes.size());
        summary.put("questions_evaluable", evaluable);
        for (int k : hits.keySet()) {
            summary.put("hit@" + k, ratio(hits.get(k), evaluable));
        }
        summary.put("mrr@10", evaluable == 0 ? null : reciprocalSum / evaluable);
        summary.put("answer_contains@10", null);
        summary.put("expected_doc_hit@10", ratio(hits.get(10), evaluable));
        summary.put("all_expected_docs_hit@10", ratio(allGoldHit10, evaluable));
        summary.put("no_gold_doc_count", noGoldDocCount);
        summary.put("invalid_extra_docs_count", 0);
        summary.put("source_filter_violation_count", sourceFilterViolationCount);
        summary.put("avg_latency_ms", latencies.stream().mapToDouble(Double::doubleValue).average().orElse(0.0d));
        summary.put("p95_latency_ms", latencies.isEmpty()
                ? 0.0d
                : latencies.get(Math.max(0, (int) Math.ceil(latencies.size() * 0.95d) - 1)));
        summary.put("avg_context_tokens", contextTokens.stream().mapToInt(Integer::intValue).average().orElse(0.0d));
        return summary;
    }

    private static Map<String, Object> groupedMetrics(
            List<Question> questions,
            List<Result> results,
            boolean bySourceType) {
        Map<String, List<Integer>> grouped = new java.util.TreeMap<>();
        for (int index = 0; index < questions.size(); index++) {
            Question question = questions.get(index);
            List<String> groups = bySourceType
                    ? (question.sourceTypes().isEmpty() ? List.of("unknown") : question.sourceTypes())
                    : List.of(question.questionType());
            for (String group : groups) {
                grouped.computeIfAbsent(group, ignored -> new ArrayList<>()).add(index);
            }
        }
        Map<String, Object> metrics = new LinkedHashMap<>();
        grouped.forEach((group, indexes) -> metrics.put(group, evaluateGroup(questions, results, indexes)));
        return metrics;
    }

    private static Double ratio(int count, int total) {
        return total == 0 ? null : count / (double) total;
    }

    private static int tokenCount(String text) {
        int count = 0;
        var matcher = TOKEN_PATTERN.matcher(text.toLowerCase());
        while (matcher.find()) {
            count++;
        }
        return count;
    }

    private static int firstRank(List<String> ranking, Set<String> expected) {
        for (int index = 0; index < ranking.size(); index++) {
            if (expected.contains(ranking.get(index))) {
                return index + 1;
            }
        }
        return -1;
    }

    private static double elapsedMs(long started) {
        return (System.nanoTime() - started) / 1_000_000.0d;
    }

    record Question(
            String id,
            String text,
            List<String> expectedDocIds,
            List<String> sourceTypes,
            String questionType) {
    }

    record RankedDocument(
            String docId,
            String chunkEsId,
            int chunkId,
            String sourceType,
            String title,
            String text,
            double score) {

        RankedDocument withScore(double newScore) {
            return new RankedDocument(docId, chunkEsId, chunkId, sourceType, title, text, newScore);
        }
    }

    record Result(
            List<RankedDocument> documents,
            double embeddingLatencyMs,
            double denseLatencyMs,
            double bm25LatencyMs,
            double keywordBm25LatencyMs,
            double englishBm25LatencyMs,
            double fusionLatencyMs,
            double latencyMs) {

        Map<String, Object> toDetails(Question question) {
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("question_id", question.id());
            details.put("question", question.text());
            details.put("question_type", question.questionType());
            details.put("source_types", question.sourceTypes());
            details.put("expected_doc_ids", question.expectedDocIds());
            details.put("evaluable", !question.expectedDocIds().isEmpty());
            List<Map<String, Object>> ranking = new ArrayList<>();
            for (int index = 0; index < documents.size(); index++) {
                RankedDocument document = documents.get(index);
                Map<String, Object> row = new HashMap<>();
                row.put("rank", index + 1);
                row.put("doc_id", document.docId());
                row.put("chunk_id", document.chunkId());
                row.put("chunk_es_id", document.chunkEsId());
                row.put("source_type", document.sourceType());
                row.put("title", document.title());
                row.put("score", document.score());
                ranking.add(row);
            }
            details.put("ranked_documents", ranking);
            details.put("embedding_latency_ms", embeddingLatencyMs);
            details.put("dense_latency_ms", denseLatencyMs);
            details.put("bm25_latency_ms", bm25LatencyMs);
            details.put("keyword_bm25_latency_ms", keywordBm25LatencyMs);
            details.put("english_bm25_latency_ms", englishBm25LatencyMs);
            details.put("fusion_latency_ms", fusionLatencyMs);
            details.put("latency_ms", latencyMs);
            return details;
        }
    }

    record Config(
            Path questions,
            Path output,
            Path detailsOutput,
            String esUrl,
            String embeddingUrl,
            String embeddingModel,
            int embeddingDimension,
            String embeddingQueryInstruction,
            String embeddingApiFormat,
            String embeddingApiKey,
            String engine,
            String index,
            String retrievalMode,
            int retrieverK,
            int denseChunkCandidates,
            int denseNumCandidates,
            int rrfK,
            double denseWeight,
            double bm25Weight,
            boolean keywordBm25Enabled,
            double keywordBm25Weight,
            boolean englishBm25Enabled,
            double englishBm25Weight,
            int topK,
            int progressEvery) {

        static Config parse(String[] args) {
            Map<String, String> values = new HashMap<>();
            for (int index = 0; index < args.length; index += 2) {
                if (index + 1 >= args.length || !args[index].startsWith("--")) {
                    throw new IllegalArgumentException("arguments must use --name value pairs");
                }
                values.put(args[index].substring(2), args[index + 1]);
            }
            return new Config(
                    Path.of(required(values, "questions")),
                    Path.of(required(values, "output")),
                    Path.of(required(values, "details-output")),
                    values.getOrDefault("es-url", "http://127.0.0.1:19200"),
                    values.getOrDefault("embedding-url", "http://127.0.0.1:18080/v1/embeddings"),
                    values.getOrDefault("embedding-model", "intfloat/multilingual-e5-small"),
                    positiveInteger(values, "embedding-dimension", 384),
                    values.getOrDefault("embedding-query-instruction", ""),
                    embeddingApiFormat(values.getOrDefault("embedding-api-format", "local")),
                    values.getOrDefault(
                            "embedding-api-key",
                            System.getenv().getOrDefault("DASHSCOPE_API_KEY", "")),
                    engine(values.getOrDefault("engine", "elasticsearch")),
                    values.getOrDefault("index", "knowledge_base_benchmark_stream_v1"),
                    retrievalMode(values.getOrDefault("retrieval-mode", "hybrid")),
                    integer(values, "retriever-k", 50),
                    integer(values, "dense-chunk-candidates", 500),
                    integer(values, "dense-num-candidates", 2500),
                    integer(values, "rrf-k", 60),
                    decimal(values, "dense-weight", 0.5d),
                    decimal(values, "bm25-weight", 1.0d),
                    bool(values, "keyword-bm25-enabled", false),
                    decimal(values, "keyword-bm25-weight", 1.25d),
                    bool(values, "english-bm25-enabled", false),
                    decimal(values, "english-bm25-weight", 1.5d),
                    integer(values, "top-k", 50),
                    integer(values, "progress-every", 25));
        }

        private static String required(Map<String, String> values, String key) {
            String value = values.get(key);
            if (value == null || value.isBlank()) {
                throw new IllegalArgumentException("missing --" + key);
            }
            return value;
        }

        private static int integer(Map<String, String> values, String key, int defaultValue) {
            return Integer.parseInt(values.getOrDefault(key, Integer.toString(defaultValue)));
        }

        private static int positiveInteger(Map<String, String> values, String key, int defaultValue) {
            int value = integer(values, key, defaultValue);
            if (value <= 0) {
                throw new IllegalArgumentException("--" + key + " must be positive");
            }
            return value;
        }

        private static double decimal(Map<String, String> values, String key, double defaultValue) {
            double value = Double.parseDouble(values.getOrDefault(key, Double.toString(defaultValue)));
            if (value <= 0.0d) {
                throw new IllegalArgumentException("--" + key + " must be positive");
            }
            return value;
        }

        private static boolean bool(Map<String, String> values, String key, boolean defaultValue) {
            String value = values.getOrDefault(key, Boolean.toString(defaultValue));
            if (!Set.of("true", "false").contains(value.toLowerCase())) {
                throw new IllegalArgumentException("--" + key + " must be true or false");
            }
            return Boolean.parseBoolean(value);
        }

        private static String retrievalMode(String value) {
            if (!Set.of("dense", "bm25", "hybrid").contains(value)) {
                throw new IllegalArgumentException("--retrieval-mode must be dense, bm25, or hybrid");
            }
            return value;
        }

        private static String embeddingApiFormat(String value) {
            if (!Set.of("local", "openai", "dashscope").contains(value)) {
                throw new IllegalArgumentException(
                        "--embedding-api-format must be local, openai, or dashscope");
            }
            return value;
        }

        private static String engine(String value) {
            if (!Set.of("elasticsearch", "opensearch").contains(value)) {
                throw new IllegalArgumentException("--engine must be elasticsearch or opensearch");
            }
            return value;
        }
    }
}
