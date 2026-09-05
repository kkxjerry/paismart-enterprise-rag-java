package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

class EnterpriseRagGenerationConsistencyTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void acceptsOnlyOneCompleteCurrentGeneration() {
        EnterpriseRagSynchronizer.CurrentState current = current(3);
        EnterpriseRagSynchronizer.ExistingState existing = existing(
                "generation-current", 3, 3, List.of("generation-current"));

        assertThat(existing.sameContent(current)).isTrue();
        assertThat(existing.canBackfillGeneration(current)).isFalse();
    }

    @Test
    void rejectsPartiallyWrittenCurrentGeneration() {
        EnterpriseRagSynchronizer.CurrentState current = current(3);
        EnterpriseRagSynchronizer.ExistingState existing = existing(
                "generation-current", 3, 1, List.of("generation-current"));

        // sameContent now includes generation completeness, not just content hashes.
        assertThat(existing.sameContent(current)).isFalse();
        assertThat(existing.canBackfillGeneration(current)).isFalse();
    }

    @Test
    void rejectsMixedOldAndCurrentGenerationsAfterInterruptedCleanup() {
        EnterpriseRagSynchronizer.CurrentState current = current(3);
        for (long observedCount : List.of(3L, 5L)) {
            EnterpriseRagSynchronizer.ExistingState existing = existing(
                    "generation-current", 3, observedCount,
                    List.of("generation-current", "generation-old"));

            assertThat(existing.sameContent(current)).as("observed count %s", observedCount).isFalse();
            assertThat(existing.canBackfillGeneration(current)).isFalse();
        }
    }

    @Test
    void rejectsMismatchedGenerationMetadataEvenWhenCountsMatch() {
        EnterpriseRagSynchronizer.CurrentState current = current(3);
        for (EnterpriseRagSynchronizer.ExistingState existing : List.of(
                existing("generation-current", 2, 3, List.of("generation-current")),
                existing("generation-old", 3, 3, List.of("generation-old")),
                existing("generation-current", 3, 3, List.of("generation-old")),
                existing("generation-current", 3, 3, List.of()))) {
            assertThat(existing.sameContent(current)).isFalse();
            assertThat(existing.canBackfillGeneration(current)).isFalse();
        }
    }

    @Test
    void permitsExactLegacyGenerationOnlyForMetadataBackfill() {
        EnterpriseRagSynchronizer.CurrentState current = current(3);
        EnterpriseRagSynchronizer.ExistingState legacy = existing("", 0, 3, List.of());
        EnterpriseRagSynchronizer.ExistingState incompleteLegacy = existing("", 0, 2, List.of());

        // Legacy content must never take the unchanged-content path.
        assertThat(legacy.sameContent(current)).isFalse();
        assertThat(legacy.canBackfillGeneration(current)).isTrue();
        assertThat(incompleteLegacy.sameContent(current)).isFalse();
        assertThat(incompleteLegacy.canBackfillGeneration(current)).isFalse();
    }

    @Test
    void rejectsLegacyBackfillWhenGenerationMetadataIsPartiallyPresent() {
        EnterpriseRagSynchronizer.CurrentState current = current(3);
        for (EnterpriseRagSynchronizer.ExistingState existing : List.of(
                existing("", 3, 3, List.of()),
                existing("generation-current", 0, 3, List.of()),
                existing("", 0, 3, List.of("generation-current")))) {
            assertThat(existing.sameContent(current)).isFalse();
            assertThat(existing.canBackfillGeneration(current)).isFalse();
        }
    }

    static Stream<Arguments> observedGenerationCases() {
        return Stream.of(
                Arguments.of("complete", false, "documents_unchanged", 0),
                Arguments.of("partial", false, "documents_reindexed", 0),
                Arguments.of("mixed", false, "documents_reindexed", 0),
                Arguments.of("legacy", false, "documents_reindexed", 0),
                Arguments.of("legacy", true, "documents_metadata_only", 1),
                Arguments.of("legacy-partial", true, "documents_reindexed", 0),
                Arguments.of("missing-aggregation", false, "documents_reindexed", 0));
    }

    @ParameterizedTest(name = "{0}, allowLegacyBackfill={1}")
    @MethodSource("observedGenerationCases")
    void parsesObservedGenerationsThroughSyncEntryPoint(
            String scenario,
            boolean allowLegacyBackfill,
            String expectedDecision,
            int expectedBackfills,
            @TempDir Path tempDir) throws Exception {
        // The parser is private now. Exercise it through main rather than restoring
        // the obsolete GenerationObservation API or invoking private methods.
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        AtomicInteger mappingRequests = new AtomicInteger();
        AtomicInteger searchRequests = new AtomicInteger();
        AtomicInteger unexpectedRequests = new AtomicInteger();
        AtomicReference<JsonNode> observedQuery = new AtomicReference<>();
        try {
            String baseUrl = "http://127.0.0.1:" + server.getAddress().getPort();
            Path docs = tempDir.resolve("docs.jsonl");
            Path aclDocs = tempDir.resolve("acl.jsonl");
            Path summary = tempDir.resolve("summary.json");
            ObjectNode document = MAPPER.createObjectNode()
                    .put("doc_id", "doc-1")
                    .put("title", "Runbook")
                    .put("text", "A runbook step with enough content for multiple chunks. ".repeat(8))
                    .put("source_type", "confluence");
            Files.writeString(docs, MAPPER.writeValueAsString(document) + "\n");
            Files.writeString(aclDocs, "");
            String[] args = {
                    "--docs", docs.toString(), "--acl-docs", aclDocs.toString(),
                    "--index", "test-index", "--es-url", baseUrl,
                    "--embedding-url", baseUrl + "/embeddings",
                    "--embedding-api-key-env", "RAG_TEST_UNUSED_API_KEY",
                    "--embedding-model", "test-model", "--embedding-dimension", "2",
                    "--chunk-size", "100", "--chunk-overlap", "0",
                    "--fail-on-missing-acl", "false", "--max-retries", "0",
                    "--dry-run", "true", "--allow-legacy-generation-backfill", Boolean.toString(allowLegacyBackfill),
                    "--sync-output", summary.toString()
            };
            EnterpriseRagSynchronizer.CurrentState current = EnterpriseRagSynchronizer.currentState(
                    document, EnterpriseRagImporter.AclDocument.empty(), EnterpriseRagImporter.Config.parse(args));
            assertThat(current.expectedChunkCount()).isGreaterThan(1);
            ObjectNode searchResponse = searchResponse(current, scenario);
            server.createContext("/", exchange -> {
                String path = exchange.getRequestURI().getPath();
                if ("GET".equals(exchange.getRequestMethod()) && "/test-index/_mapping".equals(path)) {
                    mappingRequests.incrementAndGet();
                    respond(exchange, 200, indexMapping());
                } else if ("POST".equals(exchange.getRequestMethod()) && "/test-index/_search".equals(path)) {
                    searchRequests.incrementAndGet();
                    observedQuery.set(MAPPER.readTree(exchange.getRequestBody()));
                    respond(exchange, 200, searchResponse);
                } else {
                    unexpectedRequests.incrementAndGet();
                    respond(exchange, 400, MAPPER.createObjectNode().put("error", "unexpected request"));
                }
            });
            server.start();

            EnterpriseRagSynchronizer.main(args);

            JsonNode result = MAPPER.readTree(summary.toFile());
            assertThat(result.path("dry_run").asBoolean()).isTrue();
            assertThat(result.path("documents_seen").asInt()).isEqualTo(1);
            assertThat(result.path("documents_created").asInt()).isZero();
            for (String decision : List.of("documents_unchanged", "documents_reindexed", "documents_metadata_only")) {
                assertThat(result.path(decision).asInt()).as("%s: %s", scenario, decision)
                        .isEqualTo(decision.equals(expectedDecision) ? 1 : 0);
            }
            assertThat(result.path("document_generation_backfills").asInt()).isEqualTo(expectedBackfills);
            assertThat(mappingRequests).hasValue(1);
            assertThat(searchRequests).hasValue(1);
            assertThat(unexpectedRequests).hasValue(0);
            assertThat(observedQuery.get().at("/aggs/by_document/terms/field").asText()).isEqualTo("benchmarkDocId");
            assertThat(observedQuery.get().at("/aggs/by_document/aggs/generations/terms/field").asText())
                    .isEqualTo("documentGeneration");
        } finally {
            server.stop(0);
        }
    }

    @Test
    void staleCleanupKeepsOnlyTheCurrentGeneration() {
        JsonNode body = EnterpriseRagSynchronizer.staleChunkDeleteBody("doc-1", current(3));

        assertThat(body.at("/query/bool/filter/0/term/benchmarkDocId").asText()).isEqualTo("doc-1");
        assertThat(body.at("/query/bool/must_not/0/bool/filter/0/term/documentGeneration").asText())
                .isEqualTo("generation-current");
        assertThat(body.toString()).doesNotContain("documentHash", "chunkingFingerprint", "modelVersion");
    }

    private static ObjectNode searchResponse(EnterpriseRagSynchronizer.CurrentState current, String scenario) {
        boolean legacy = scenario.startsWith("legacy");
        ObjectNode source = MAPPER.valueToTree(current);
        source.remove(List.of("expectedChunkCount", "acl"));
        source.put("benchmarkDocId", "doc-1");
        if (legacy) {
            source.remove(List.of("documentGeneration", "documentChunkCount"));
        } else {
            source.put("documentChunkCount", current.expectedChunkCount());
        }
        ObjectNode response = MAPPER.createObjectNode();
        response.putObject("hits").putArray("hits").addObject().set("_source", source);
        if (!"missing-aggregation".equals(scenario)) {
            int count = current.expectedChunkCount()
                    - ("partial".equals(scenario) || "legacy-partial".equals(scenario) ? 1 : 0);
            ObjectNode bucket = response.putObject("aggregations").putObject("by_document")
                    .putArray("buckets").addObject().put("key", "doc-1").put("doc_count", count);
            ArrayNode generations = bucket.putObject("generations").putArray("buckets");
            if (!legacy) {
                generations.addObject().put("key", current.documentGeneration())
                        .put("doc_count", "mixed".equals(scenario) ? count - 1 : count);
                if ("mixed".equals(scenario)) {
                    generations.addObject().put("key", "generation-old").put("doc_count", 1);
                }
            }
        }
        return response;
    }

    private static ObjectNode indexMapping() {
        ObjectNode response = MAPPER.createObjectNode();
        ObjectNode properties = response.putObject("test-index").putObject("mappings").putObject("properties");
        properties.putObject("vector").put("type", "dense_vector").put("dims", 2);
        properties.putObject("textContent").put("type", "text").put("analyzer", "standard");
        for (String field : List.of("documentVersion", "documentHash", "documentGeneration", "contentHash", "aclHash",
                "chunkKind", "chunkingStrategy", "chunkingFingerprint", "sourceRevision")) {
            properties.putObject(field).put("type", "keyword");
        }
        properties.putObject("documentChunkCount").put("type", "integer");
        for (String field : List.of("sourceUpdatedAt", "eventTime", "deletedAt")) {
            properties.putObject(field).put("type", "date");
        }
        return response;
    }

    private static void respond(HttpExchange exchange, int status, JsonNode body) throws IOException {
        try (exchange) {
            byte[] bytes = MAPPER.writeValueAsBytes(body);
            exchange.getResponseHeaders().set("Content-Type", "application/json; charset=" + StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(status, bytes.length);
            exchange.getResponseBody().write(bytes);
        }
    }

    private static EnterpriseRagSynchronizer.CurrentState current(int chunkCount) {
        EnterpriseRagImporter.AclDocument acl = new EnterpriseRagImporter.AclDocument(
                "tenant-a", "internal", List.of("team-a"), List.of());
        return new EnterpriseRagSynchronizer.CurrentState(
                "document-hash", acl.hash(), "source-aware:1200:200", "test-model",
                "generation-current", chunkCount, "confluence", "confluence:runbook", "dataset",
                "v2", "2026-09-03T00:00:00Z", "revision-2", acl);
    }

    private static EnterpriseRagSynchronizer.ExistingState existing(
            String generation,
            int expectedCount,
            long observedCount,
            List<String> observedGenerations) {
        EnterpriseRagSynchronizer.CurrentState current = current(3);
        return new EnterpriseRagSynchronizer.ExistingState(
                current.documentHash(), current.aclHash(), current.chunkingFingerprint(), current.modelVersion(),
                generation, expectedCount, observedCount, observedGenerations,
                current.sourceType(), current.sourcePath(), current.sourceDataset(),
                current.documentVersion(), current.sourceUpdatedAt(), current.sourceRevision());
    }
}
