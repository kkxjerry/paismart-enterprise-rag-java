package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.BufferedReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Duration;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Incremental, fail-closed synchronization for an EnterpriseRAG index.
 *
 * <p>It skips embeddings for unchanged content, updates ACL/source metadata in
 * place, removes stale chunks after a successful replacement, and can propagate
 * source deletions within the managed tenant set.</p>
 */
public final class EnterpriseRagSynchronizer {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private EnterpriseRagSynchronizer() {
    }

    public static void main(String[] args) throws Exception {
        EnterpriseRagImporter.Config importer = EnterpriseRagImporter.Config.parse(args);
        Arguments values = Arguments.parse(args);
        boolean deleteMissing = values.bool("delete-missing", false);
        Set<String> managedTenants = managedTenantSet(values.string("managed-tenants", ""));
        SyncConfig config = new SyncConfig(
                importer,
                deleteMissing,
                values.bool("dry-run", false),
                values.bool("allow-legacy-generation-backfill", false),
                values.path("sync-output", "runs/index-sync-summary.json"),
                managedTenants);
        if (importer.maxDocuments() > 0 && config.deleteMissing()) {
            throw new IllegalArgumentException("--delete-missing cannot be combined with --max-documents");
        }

        Map<String, EnterpriseRagImporter.AclDocument> aclByDocId = EnterpriseRagImporter.loadAcl(importer.aclDocs());
        if (importer.failOnMissingAcl() && aclByDocId.isEmpty()) {
            throw new IllegalArgumentException("ACL input is empty while fail-on-missing-acl is enabled");
        }
        Set<String> sourceTenants = new HashSet<>();
        aclByDocId.values().stream()
                .map(EnterpriseRagImporter.AclDocument::tenantId)
                .filter(value -> !value.isBlank())
                .forEach(sourceTenants::add);
        // A destructive delete-missing run must prove its tenant scope before
        // contacting Elasticsearch. Misconfiguration therefore fails before
        // any index read or write, even when the target mapping is stale.
        validateDeletionScope(config.deleteMissing(), config.managedTenants(), sourceTenants);

        JsonHttpClient http = new JsonHttpClient();
        EnterpriseRagImporter.validateIndex(http, importer);
        EmbeddingClient embeddings = new EmbeddingClient(importer.embeddingConfig());
        Counters counters = new Counters();
        Set<String> sourceDocIds = new HashSet<>();
        long started = System.nanoTime();

        List<JsonNode> batch = new ArrayList<>(importer.documentBatchSize());
        try (BufferedReader reader = Files.newBufferedReader(importer.docs(), StandardCharsets.UTF_8)) {
            String line;
            int lineNumber = 0;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.isBlank()) {
                    continue;
                }
                JsonNode document;
                try {
                    document = MAPPER.readTree(line);
                } catch (Exception exception) {
                    throw new IllegalArgumentException("invalid JSONL at " + importer.docs() + ":" + lineNumber, exception);
                }
                String docId = requiredText(document, "doc_id");
                sourceDocIds.add(docId);
                batch.add(document);
                boolean limitReached = importer.maxDocuments() > 0
                        && counters.documentsSeen + batch.size() >= importer.maxDocuments();
                if (batch.size() >= importer.documentBatchSize() || limitReached) {
                    synchronizeBatch(config, batch, aclByDocId, embeddings, http, counters);
                    printProgress(counters, started);
                    batch.clear();
                }
                if (limitReached) {
                    break;
                }
            }
        }
        if (!batch.isEmpty()) {
            synchronizeBatch(config, batch, aclByDocId, embeddings, http, counters);
            printProgress(counters, started);
        }

        if (config.deleteMissing()) {
            Set<String> indexed = scanIndexedDocumentIds(http, importer, config.managedTenants());
            indexed.removeAll(sourceDocIds);
            counters.documentsMissingFromSource = indexed.size();
            if (!config.dryRun()) {
                counters.documentsDeleted = deleteDocuments(http, importer, indexed, config.managedTenants());
            }
        }
        if (!config.dryRun()) {
            http.requireJson(
                    "POST",
                    indexUrl(importer) + "/_refresh",
                    null,
                    "",
                    importer.maxRetries(),
                    Duration.ofSeconds(60));
        }

        ObjectNode summary = counters.toJson();
        summary.put("event", "index_sync_complete");
        summary.put("index", importer.index());
        summary.put("dry_run", config.dryRun());
        summary.put("delete_missing", config.deleteMissing());
        summary.put("allow_legacy_generation_backfill", config.allowLegacyGenerationBackfill());
        summary.set("managed_tenants", MAPPER.valueToTree(config.managedTenants().stream().sorted().toList()));
        summary.put("chunking_strategy", importer.chunkingStrategy());
        summary.set("source_aware_types", MAPPER.valueToTree(
                importer.sourceAwareTypes().stream().sorted().toList()));
        summary.put("chunking_fingerprint", chunkingFingerprint(importer));
        summary.put("elapsed_seconds", elapsedMs(started) / 1000.0d);
        summary.put("completed_at", OffsetDateTime.now(ZoneOffset.UTC).toString());
        Path output = config.output().toAbsolutePath().normalize();
        Path parent = output.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        MAPPER.writerWithDefaultPrettyPrinter().writeValue(output.toFile(), summary);
        System.out.println(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(summary));
    }

    private static void synchronizeBatch(
            SyncConfig sync,
            List<JsonNode> documents,
            Map<String, EnterpriseRagImporter.AclDocument> aclByDocId,
            EmbeddingClient embeddings,
            JsonHttpClient http,
            Counters counters) throws Exception {
        EnterpriseRagImporter.Config config = sync.importer();
        Map<String, ExistingState> existing = loadExistingStates(
                http,
                config,
                documents.stream().map(document -> requiredText(document, "doc_id")).toList());
        List<JsonNode> reindex = new ArrayList<>();
        List<MetadataUpdate> metadataUpdates = new ArrayList<>();
        Map<String, CurrentState> currentByDoc = new LinkedHashMap<>();

        for (JsonNode document : documents) {
            String docId = requiredText(document, "doc_id");
            EnterpriseRagImporter.AclDocument acl = aclByDocId.get(docId);
            if (acl == null) {
                if (config.failOnMissingAcl()) {
                    throw new IllegalArgumentException("missing ACL for document " + docId);
                }
                acl = EnterpriseRagImporter.AclDocument.empty();
            }
            CurrentState current = currentState(document, acl, config);
            currentByDoc.put(docId, current);
            ExistingState prior = existing.get(docId);
            counters.documentsSeen++;
            if (prior == null) {
                reindex.add(document);
                counters.documentsCreated++;
            } else if (prior.sameContent(current)) {
                if (prior.sameMetadata(current)) {
                    counters.documentsUnchanged++;
                } else {
                    metadataUpdates.add(new MetadataUpdate(docId, current));
                    counters.documentsMetadataOnly++;
                }
            } else if (sync.allowLegacyGenerationBackfill() && prior.canBackfillGeneration(current)) {
                metadataUpdates.add(new MetadataUpdate(docId, current));
                counters.documentsMetadataOnly++;
                counters.documentGenerationBackfills++;
            } else {
                reindex.add(document);
                counters.documentsReindexed++;
            }
        }

        if (sync.dryRun()) {
            return;
        }
        for (MetadataUpdate update : metadataUpdates) {
            updateMetadata(http, config, update);
        }
        if (reindex.isEmpty()) {
            return;
        }
        List<EnterpriseRagImporter.Chunk> chunks = EnterpriseRagImporter.buildChunks(config, reindex, aclByDocId);
        List<String> passages = chunks.stream()
                .map(chunk -> chunk.source().path("title").asText() + "\n" + chunk.source().path("textContent").asText())
                .toList();
        List<List<Double>> vectors = EnterpriseRagImporter.embedAll(config, embeddings, passages);
        if (vectors.size() != chunks.size()) {
            throw new IllegalStateException("embedding count does not match incremental chunk count");
        }
        for (int index = 0; index < chunks.size(); index++) {
            chunks.get(index).source().set("vector", MAPPER.valueToTree(vectors.get(index)));
        }
        EnterpriseRagImporter.bulkIndex(config, http, chunks);
        counters.chunksIndexed += chunks.size();
        for (JsonNode document : reindex) {
            String docId = requiredText(document, "doc_id");
            counters.staleChunksDeleted += deleteStaleChunks(http, config, docId, currentByDoc.get(docId));
        }
    }

    static CurrentState currentState(
            JsonNode document,
            EnterpriseRagImporter.AclDocument acl,
            EnterpriseRagImporter.Config config) {
        String title = document.path("title").asText("");
        String text = document.path("text").asText("");
        String sourceType = document.path("source_type").asText("unknown");
        String documentHash = sha256(title + "\n" + text);
        String chunkingFingerprint = chunkingFingerprint(config);
        int expectedChunkCount = EnterpriseRagImporter.segmentsForDocument(
                config,
                sourceType,
                title,
                text).size();
        String documentGeneration = EnterpriseRagImporter.documentGeneration(
                documentHash,
                chunkingFingerprint,
                config.embeddingModel(),
                sourceType);
        return new CurrentState(
                documentHash,
                acl.hash(),
                chunkingFingerprint,
                config.embeddingModel(),
                documentGeneration,
                expectedChunkCount,
                sourceType,
                document.path("source_path").asText(""),
                document.path("source_dataset").asText(""),
                firstText(
                        document.path("document_version"),
                        document.path("version"),
                        document.path("metadata").path("document_version"),
                        document.path("metadata").path("version")),
                firstText(
                        document.path("source_updated_at"),
                        document.path("updated_at"),
                        document.path("metadata").path("source_updated_at"),
                        document.path("metadata").path("updated_at")),
                firstText(
                        document.path("source_revision"),
                        document.path("revision"),
                        document.path("metadata").path("source_revision"),
                        document.path("metadata").path("revision")),
                acl);
    }

    static ObjectNode existingStateQuery(List<String> docIds) {
        ObjectNode body = MAPPER.createObjectNode();
        body.put("size", Math.max(1, docIds.size()));
        body.put("track_total_hits", false);
        body.putArray("_source")
                .add("benchmarkDocId")
                .add("documentHash")
                .add("aclHash")
                .add("chunkingFingerprint")
                .add("modelVersion")
                .add("documentGeneration")
                .add("documentChunkCount")
                .add("sourceType")
                .add("sourcePath")
                .add("sourceDataset")
                .add("documentVersion")
                .add("sourceUpdatedAt")
                .add("sourceRevision");
        body.set("query", termsQuery("benchmarkDocId", docIds));
        body.putObject("collapse").put("field", "benchmarkDocId");
        body.putArray("sort").addObject().put("indexedAt", "desc");
        ObjectNode byDocumentAggregation = body.putObject("aggs").putObject("by_document");
        ObjectNode byDocument = byDocumentAggregation.putObject("terms");
        byDocument.put("field", "benchmarkDocId");
        byDocument.put("size", Math.max(1, docIds.size()));
        byDocumentAggregation.putObject("aggs")
                .putObject("generations")
                .putObject("terms")
                .put("field", "documentGeneration")
                .put("size", 4);
        return body;
    }

    private static Map<String, ExistingState> loadExistingStates(
            JsonHttpClient http,
            EnterpriseRagImporter.Config config,
            List<String> docIds) throws Exception {
        JsonNode response = http.requireJson(
                "POST",
                indexUrl(config) + "/_search",
                existingStateQuery(docIds),
                "",
                config.maxRetries(),
                Duration.ofSeconds(60));
        Map<String, Long> actualChunkCounts = new LinkedHashMap<>();
        Map<String, List<String>> generationsByDoc = new LinkedHashMap<>();
        for (JsonNode bucket : response.path("aggregations").path("by_document").path("buckets")) {
            String docId = bucket.path("key").asText();
            if (docId.isBlank()) {
                continue;
            }
            actualChunkCounts.put(docId, bucket.path("doc_count").asLong(0L));
            List<String> generations = new ArrayList<>();
            for (JsonNode generation : bucket.path("generations").path("buckets")) {
                String value = generation.path("key").asText();
                if (!value.isBlank()) {
                    generations.add(value);
                }
            }
            generationsByDoc.put(docId, List.copyOf(generations));
        }

        Map<String, ExistingState> values = new LinkedHashMap<>();
        for (JsonNode hit : response.path("hits").path("hits")) {
            JsonNode source = hit.path("_source");
            String docId = source.path("benchmarkDocId").asText();
            if (!docId.isBlank()) {
                values.put(docId, new ExistingState(
                        source.path("documentHash").asText(),
                        source.path("aclHash").asText(),
                        source.path("chunkingFingerprint").asText(),
                        source.path("modelVersion").asText(),
                        source.path("documentGeneration").asText(),
                        source.path("documentChunkCount").asInt(0),
                        actualChunkCounts.getOrDefault(docId, 0L),
                        generationsByDoc.getOrDefault(docId, List.of()),
                        source.path("sourceType").asText(),
                        source.path("sourcePath").asText(),
                        source.path("sourceDataset").asText(),
                        source.path("documentVersion").asText(),
                        source.path("sourceUpdatedAt").asText(),
                        source.path("sourceRevision").asText()));
            }
        }
        return Map.copyOf(values);
    }

    static ObjectNode metadataUpdateBody(MetadataUpdate update) {
        CurrentState state = update.state();
        ObjectNode body = MAPPER.createObjectNode();
        ObjectNode script = body.putObject("script");
        script.put("lang", "painless");
        script.put("source", """
                ctx._source.tenantId = params.tenantId;
                ctx._source.classification = params.classification;
                ctx._source.allowedGroupIds = params.allowedGroupIds;
                ctx._source.deniedGroupIds = params.deniedGroupIds;
                ctx._source.aclHash = params.aclHash;
                ctx._source.sourceType = params.sourceType;
                ctx._source.sourcePath = params.sourcePath;
                ctx._source.sourceDataset = params.sourceDataset;
                ctx._source.documentGeneration = params.documentGeneration;
                ctx._source.documentChunkCount = params.documentChunkCount;
                if (params.documentVersion == '') { ctx._source.remove('documentVersion'); }
                else { ctx._source.documentVersion = params.documentVersion; }
                if (params.sourceUpdatedAt == '') { ctx._source.remove('sourceUpdatedAt'); }
                else { ctx._source.sourceUpdatedAt = params.sourceUpdatedAt; }
                if (params.sourceRevision == '') { ctx._source.remove('sourceRevision'); }
                else { ctx._source.sourceRevision = params.sourceRevision; }
                """);
        ObjectNode params = script.putObject("params");
        params.put("tenantId", state.acl().tenantId());
        params.put("classification", state.acl().classification());
        params.set("allowedGroupIds", MAPPER.valueToTree(state.acl().allowedGroupIds()));
        params.set("deniedGroupIds", MAPPER.valueToTree(state.acl().deniedGroupIds()));
        params.put("aclHash", state.aclHash());
        params.put("sourceType", state.sourceType());
        params.put("sourcePath", state.sourcePath());
        params.put("sourceDataset", state.sourceDataset());
        params.put("documentGeneration", state.documentGeneration());
        params.put("documentChunkCount", state.expectedChunkCount());
        params.put("documentVersion", state.documentVersion());
        params.put("sourceUpdatedAt", state.sourceUpdatedAt());
        params.put("sourceRevision", state.sourceRevision());
        body.set("query", termQuery("benchmarkDocId", update.docId()));
        return body;
    }

    private static void updateMetadata(
            JsonHttpClient http,
            EnterpriseRagImporter.Config config,
            MetadataUpdate update) throws Exception {
        http.requireJson(
                "POST",
                indexUrl(config) + "/_update_by_query?conflicts=proceed&refresh=false",
                metadataUpdateBody(update),
                "",
                config.maxRetries(),
                Duration.ofMinutes(2));
    }

    static ObjectNode staleChunkDeleteBody(String docId, CurrentState state) {
        ObjectNode current = MAPPER.createObjectNode();
        ArrayNode filters = current.putArray("filter");
        filters.add(termQuery("documentGeneration", state.documentGeneration()));

        ObjectNode bool = MAPPER.createObjectNode();
        bool.putArray("filter").add(termQuery("benchmarkDocId", docId));
        bool.putArray("must_not").add(MAPPER.createObjectNode().set("bool", current));
        return MAPPER.createObjectNode().set("query", MAPPER.createObjectNode().set("bool", bool));
    }

    private static long deleteStaleChunks(
            JsonHttpClient http,
            EnterpriseRagImporter.Config config,
            String docId,
            CurrentState state) throws Exception {
        JsonNode response = http.requireJson(
                "POST",
                indexUrl(config) + "/_delete_by_query?conflicts=proceed&refresh=false",
                staleChunkDeleteBody(docId, state),
                "",
                config.maxRetries(),
                Duration.ofMinutes(2));
        return response.path("deleted").asLong(0L);
    }

    static Set<String> scanIndexedDocumentIds(
            JsonHttpClient http,
            EnterpriseRagImporter.Config config,
            Set<String> managedTenants) throws Exception {
        Set<String> values = new HashSet<>();
        ObjectNode after = null;
        do {
            ObjectNode body = MAPPER.createObjectNode();
            body.put("size", 0);
            if (!managedTenants.isEmpty()) {
                body.set("query", termsQuery("tenantId", managedTenants.stream().sorted().toList()));
            }
            ObjectNode composite = body.putObject("aggs").putObject("documents").putObject("composite");
            composite.put("size", 1_000);
            composite.putArray("sources")
                    .addObject()
                    .putObject("doc_id")
                    .putObject("terms")
                    .put("field", "benchmarkDocId");
            if (after != null) {
                composite.set("after", after);
            }
            JsonNode response = http.requireJson(
                    "POST",
                    indexUrl(config) + "/_search",
                    body,
                    "",
                    config.maxRetries(),
                    Duration.ofMinutes(2));
            JsonNode aggregation = response.path("aggregations").path("documents");
            for (JsonNode bucket : aggregation.path("buckets")) {
                String docId = bucket.path("key").path("doc_id").asText();
                if (!docId.isBlank()) {
                    values.add(docId);
                }
            }
            after = aggregation.path("after_key").isObject()
                    ? (ObjectNode) aggregation.path("after_key").deepCopy()
                    : null;
        } while (after != null);
        return values;
    }

    private static long deleteDocuments(
            JsonHttpClient http,
            EnterpriseRagImporter.Config config,
            Set<String> docIds,
            Set<String> managedTenants) throws Exception {
        long deleted = 0L;
        List<String> ordered = docIds.stream().sorted().toList();
        for (int start = 0; start < ordered.size(); start += 500) {
            List<String> batch = ordered.subList(start, Math.min(start + 500, ordered.size()));
            ObjectNode bool = MAPPER.createObjectNode();
            ArrayNode filter = bool.putArray("filter");
            filter.add(termsQuery("benchmarkDocId", batch));
            if (!managedTenants.isEmpty()) {
                filter.add(termsQuery("tenantId", managedTenants.stream().sorted().toList()));
            }
            ObjectNode body = MAPPER.createObjectNode();
            body.set("query", MAPPER.createObjectNode().set("bool", bool));
            JsonNode response = http.requireJson(
                    "POST",
                    indexUrl(config) + "/_delete_by_query?conflicts=proceed&refresh=false",
                    body,
                    "",
                    config.maxRetries(),
                    Duration.ofMinutes(5));
            deleted += response.path("deleted").asLong(0L);
        }
        return deleted;
    }

    static String chunkingFingerprint(EnterpriseRagImporter.Config config) {
        return EnterpriseRagImporter.chunkingFingerprint(config);
    }

    private static ObjectNode termQuery(String field, String value) {
        return MAPPER.createObjectNode().set("term", MAPPER.createObjectNode().put(field, value));
    }

    private static ObjectNode termsQuery(String field, List<String> values) {
        return MAPPER.createObjectNode().set(
                "terms",
                MAPPER.createObjectNode().set(field, MAPPER.valueToTree(values)));
    }

    private static String indexUrl(EnterpriseRagImporter.Config config) {
        String base = config.esUrl().endsWith("/")
                ? config.esUrl().substring(0, config.esUrl().length() - 1)
                : config.esUrl();
        return base + "/" + config.index();
    }

    static void validateDeletionScope(
            boolean deleteMissing,
            Set<String> managedTenants,
            Set<String> sourceTenants) {
        if (!deleteMissing) {
            return;
        }
        if (managedTenants == null || managedTenants.isEmpty()) {
            throw new IllegalArgumentException(
                    "--managed-tenants is required with --delete-missing to prevent cross-tenant deletion");
        }
        if (sourceTenants != null && !managedTenants.containsAll(sourceTenants)) {
            Set<String> unmanaged = new HashSet<>(sourceTenants);
            unmanaged.removeAll(managedTenants);
            throw new IllegalArgumentException(
                    "ACL input contains tenants outside --managed-tenants: " + unmanaged);
        }
    }

    static Set<String> managedTenantSet(String raw) {
        if (raw == null || raw.isBlank()) {
            return Set.of();
        }
        Set<String> values = new HashSet<>();
        for (String value : raw.split(",")) {
            String tenant = value.trim();
            if (!tenant.isBlank()) {
                values.add(tenant);
            }
        }
        return Set.copyOf(values);
    }

    private static String requiredText(JsonNode node, String field) {
        String value = node.path(field).asText();
        if (value.isBlank()) {
            throw new IllegalArgumentException("document is missing " + field);
        }
        return value;
    }

    private static String firstText(JsonNode... values) {
        for (JsonNode value : values) {
            if (value != null && value.isValueNode() && !value.asText().isBlank()) {
                return value.asText();
            }
        }
        return "";
    }

    private static String sha256(String text) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(digest.digest(text.getBytes(StandardCharsets.UTF_8)));
        } catch (java.security.NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 unavailable", exception);
        }
    }

    private static void printProgress(Counters counters, long started) throws Exception {
        ObjectNode row = counters.toJson();
        row.put("event", "index_sync_progress");
        row.put("elapsed_ms", elapsedMs(started));
        System.out.println(MAPPER.writeValueAsString(row));
    }

    private static double elapsedMs(long started) {
        return (System.nanoTime() - started) / 1_000_000.0d;
    }

    record SyncConfig(
            EnterpriseRagImporter.Config importer,
            boolean deleteMissing,
            boolean dryRun,
            boolean allowLegacyGenerationBackfill,
            Path output,
            Set<String> managedTenants) {

        SyncConfig {
            managedTenants = managedTenants == null ? Set.of() : Set.copyOf(managedTenants);
        }
    }

    record CurrentState(
            String documentHash,
            String aclHash,
            String chunkingFingerprint,
            String modelVersion,
            String documentGeneration,
            int expectedChunkCount,
            String sourceType,
            String sourcePath,
            String sourceDataset,
            String documentVersion,
            String sourceUpdatedAt,
            String sourceRevision,
            EnterpriseRagImporter.AclDocument acl) {
    }

    record ExistingState(
            String documentHash,
            String aclHash,
            String chunkingFingerprint,
            String modelVersion,
            String documentGeneration,
            int declaredChunkCount,
            long actualChunkCount,
            List<String> observedGenerations,
            String sourceType,
            String sourcePath,
            String sourceDataset,
            String documentVersion,
            String sourceUpdatedAt,
            String sourceRevision) {

        ExistingState {
            observedGenerations = observedGenerations == null
                    ? List.of()
                    : observedGenerations.stream().filter(value -> !value.isBlank()).distinct().toList();
        }

        boolean sameContent(CurrentState current) {
            return sameBaseContent(current)
                    && completeGeneration(current);
        }

        boolean canBackfillGeneration(CurrentState current) {
            return sameBaseContent(current)
                    && actualChunkCount == current.expectedChunkCount()
                    && observedGenerations.isEmpty()
                    && documentGeneration.isBlank()
                    && declaredChunkCount == 0;
        }

        private boolean sameBaseContent(CurrentState current) {
            return documentHash.equals(current.documentHash())
                    && chunkingFingerprint.equals(current.chunkingFingerprint())
                    && modelVersion.equals(current.modelVersion())
                    && sourceType.equals(current.sourceType());
        }

        private boolean completeGeneration(CurrentState current) {
            return declaredChunkCount == current.expectedChunkCount()
                    && actualChunkCount == current.expectedChunkCount()
                    && documentGeneration.equals(current.documentGeneration())
                    && observedGenerations.equals(List.of(current.documentGeneration()));
        }

        boolean sameMetadata(CurrentState current) {
            return aclHash.equals(current.aclHash())
                    && sourcePath.equals(current.sourcePath())
                    && sourceDataset.equals(current.sourceDataset())
                    && documentVersion.equals(current.documentVersion())
                    && sourceUpdatedAt.equals(current.sourceUpdatedAt())
                    && sourceRevision.equals(current.sourceRevision());
        }
    }

    record MetadataUpdate(String docId, CurrentState state) {
    }

    static final class Counters {
        long documentsSeen;
        long documentsCreated;
        long documentsReindexed;
        long documentsMetadataOnly;
        long documentGenerationBackfills;
        long documentsUnchanged;
        long chunksIndexed;
        long staleChunksDeleted;
        long documentsMissingFromSource;
        long documentsDeleted;

        ObjectNode toJson() {
            ObjectNode row = MAPPER.createObjectNode();
            row.put("documents_seen", documentsSeen);
            row.put("documents_created", documentsCreated);
            row.put("documents_reindexed", documentsReindexed);
            row.put("documents_metadata_only", documentsMetadataOnly);
            row.put("document_generation_backfills", documentGenerationBackfills);
            row.put("documents_unchanged", documentsUnchanged);
            row.put("chunks_indexed", chunksIndexed);
            row.put("stale_chunks_deleted", staleChunksDeleted);
            row.put("documents_missing_from_source", documentsMissingFromSource);
            row.put("documents_deleted", documentsDeleted);
            return row;
        }
    }
}
