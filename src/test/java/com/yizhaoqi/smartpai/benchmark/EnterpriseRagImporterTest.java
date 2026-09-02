package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class EnterpriseRagImporterTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void writesDocumentVersionAndHashesIntoEveryEvidenceChunk() {
        EnterpriseRagImporter.Config config = EnterpriseRagImporter.Config.parse(new String[] {
                "--docs", "docs.jsonl",
                "--acl-docs", "acl.jsonl",
                "--index", "benchmark_v2",
                "--chunk-size", "1200",
                "--chunk-overlap", "200"
        });
        ObjectNode document = MAPPER.createObjectNode();
        document.put("doc_id", "doc-1");
        document.put("title", "Contractor access policy");
        document.put("text", "Contractor access expires after 90 days by default.");
        document.put("source_type", "confluence");
        document.put("source_path", "confluence:access-policy");
        document.put("document_version", "v2");
        document.put("source_updated_at", "2026-08-17T12:00:00Z");
        EnterpriseRagImporter.AclDocument acl = new EnterpriseRagImporter.AclDocument(
                "tenant_redwood",
                "internal",
                List.of("source:confluence"),
                List.of());

        List<EnterpriseRagImporter.Chunk> chunks = EnterpriseRagImporter.buildChunks(
                config,
                List.of(document),
                Map.of("doc-1", acl));

        assertThat(chunks).hasSize(1);
        ObjectNode source = chunks.get(0).source();
        assertThat(source.path("documentVersion").asText()).isEqualTo("v2");
        assertThat(source.path("sourceUpdatedAt").asText()).isEqualTo("2026-08-17T12:00:00Z");
        assertThat(source.path("documentHash").asText()).hasSize(64);
        assertThat(source.path("contentHash").asText()).hasSize(64);
        assertThat(source.path("documentHash").asText())
                .isNotEqualTo(source.path("contentHash").asText());
        assertThat(source.path("classification").asText()).isEqualTo("internal");
    }

    @Test
    void acceptsEvidenceSchemaAndRejectsTheHistoricalStrictMappingForNewImports() throws Exception {
        var evidenceMapping = MAPPER.readTree(
                Path.of("config/elasticsearch-evidence-2048.json").toFile()).path("mappings");
        var historicalMapping = MAPPER.readTree(
                Path.of("config/elasticsearch-2048.json").toFile()).path("mappings");

        assertThatCode(() -> EnterpriseRagImporter.validateMapping(evidenceMapping, 2048))
                .doesNotThrowAnyException();
        assertThatThrownBy(() -> EnterpriseRagImporter.validateMapping(historicalMapping, 2048))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("documentVersion");
    }

    @Test
    void checkpointSignatureTreatsEquivalentJsonNumbersAsEqual() throws Exception {
        Map<String, Object> signature = new LinkedHashMap<>();
        signature.put("docs_size", 404L);
        signature.put("docs_mtime_ms", 1_787_000_000_000L);
        signature.put("index", "benchmark_v1");
        boolean matches = EnterpriseRagImporter.signatureMatches(
                MAPPER.readTree("{\"docs_size\":404,\"docs_mtime_ms\":1787000000000,\"index\":\"benchmark_v1\"}"),
                signature);

        assertThat(matches).isTrue();
    }
}
