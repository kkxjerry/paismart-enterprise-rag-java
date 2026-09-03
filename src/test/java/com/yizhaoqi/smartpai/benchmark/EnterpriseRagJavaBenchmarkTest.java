package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yizhaoqi.smartpai.service.EvidenceBuilder;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class EnterpriseRagJavaBenchmarkTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void sourceAclFilterRequiresTenantSourceAndAllowedGroup() {
        JsonNode filter = EnterpriseRagJavaBenchmark.sourceAclFilter(List.of("confluence"));
        String json = filter.toString();

        assertThat(json).contains("tenant_redwood");
        assertThat(json).contains("sourceType");
        assertThat(json).contains("confluence");
        assertThat(json).contains("source:confluence");
        assertThat(json).contains("deniedGroupIds");
    }

    @Test
    void noSourceDoesNotInventAnAclScope() {
        assertThat(EnterpriseRagJavaBenchmark.sourceAclFilter(List.of())).isNull();
    }

    @Test
    void parsesWeightedHybridDefaults() {
        EnterpriseRagJavaBenchmark.Config config = EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                "--questions", "questions.json",
                "--output", "summary.json",
                "--details-output", "details.jsonl"
        });

        assertThat(config.denseWeight()).isEqualTo(0.5d);
        assertThat(config.bm25Weight()).isEqualTo(1.0d);
        assertThat(config.rrfK()).isEqualTo(60);
        assertThat(config.engine()).isEqualTo("elasticsearch");
        assertThat(config.embeddingModel()).isEqualTo("intfloat/multilingual-e5-small");
        assertThat(config.embeddingDimension()).isEqualTo(384);
        assertThat(config.embeddingQueryInstruction()).isEmpty();
        assertThat(config.embeddingApiFormat()).isEqualTo("local");
        assertThat(config.keywordBm25Enabled()).isFalse();
        assertThat(config.englishBm25Enabled()).isFalse();
    }

    @Test
    void rejectsMisspelledExperimentArgumentsInsteadOfSilentlyUsingDefaults() {
        assertThatThrownBy(() -> EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                "--questions", "questions.json",
                "--output", "summary.json",
                "--details-output", "details.jsonl",
                "--dense-weigth", "0.75"
        }))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("unknown evaluate arguments")
                .hasMessageContaining("dense-weigth");
    }

    @Test
    void normalizesEvidenceDepthAndRejectsOutputPathCollisions() {
        EnterpriseRagJavaBenchmark.Config config = EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                "--questions", "questions.json",
                "--output", "summary.json",
                "--details-output", "details.jsonl",
                "--top-k", "5",
                "--evidence-enabled", "true",
                "--evidence-top-documents", "10"
        });

        assertThat(config.evidenceTopDocuments()).isEqualTo(5);
        assertThatThrownBy(() -> EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                "--questions", "questions.json",
                "--output", "same.json",
                "--details-output", "same.json"
        }))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("distinct paths");
    }

    @Test
    void rejectsOutputsThatOverwriteTheExperimentConfigOrDeclaredInputs(@TempDir Path tempDir) throws Exception {
        Path questions = tempDir.resolve("questions.json");
        Path documents = tempDir.resolve("docs.jsonl");
        Files.writeString(questions, "[]");
        Files.writeString(documents, "{}\n");

        Path inputCollision = tempDir.resolve("input-collision.json");
        Files.writeString(inputCollision, """
                {
                  "schema_version": 1,
                  "arguments": {
                    "questions": "questions.json",
                    "output": "docs.jsonl",
                    "details-output": "details.jsonl"
                  },
                  "inputs": {
                    "documents": "docs.jsonl"
                  }
                }
                """);
        assertThatThrownBy(() -> EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                "--config", inputCollision.toString()
        }))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("must not overwrite")
                .hasMessageContaining("docs.jsonl");

        Path configCollision = tempDir.resolve("config-collision.json");
        Files.writeString(configCollision, """
                {
                  "schema_version": 1,
                  "arguments": {
                    "questions": "questions.json",
                    "output": "config-collision.json",
                    "details-output": "details.jsonl"
                  }
                }
                """);
        assertThatThrownBy(() -> EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                "--config", configCollision.toString()
        }))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("must not overwrite")
                .hasMessageContaining("config-collision.json");
    }

    @Test
    void buildsEmbeddingRequestFromConfigInsteadOfHardCodingE5Small() {
        EnterpriseRagJavaBenchmark.Config config = EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                "--questions", "questions.json",
                "--output", "summary.json",
                "--details-output", "details.jsonl",
                "--embedding-model", "intfloat/multilingual-e5-base",
                "--embedding-dimension", "768",
                "--embedding-query-instruction", "Retrieve relevant enterprise passages"
        });

        JsonNode body = EnterpriseRagJavaBenchmark.embeddingRequestBody(config, "How long is access valid?");

        assertThat(body.path("model").asText()).isEqualTo("intfloat/multilingual-e5-base");
        assertThat(body.path("dimension").asInt()).isEqualTo(768);
        assertThat(body.path("input_type").asText()).isEqualTo("query");
        assertThat(body.path("input").path(0).asText()).isEqualTo(
                "Instruct: Retrieve relevant enterprise passages\nQuery:How long is access valid?");
    }

    @Test
    void buildsOpenAiCompatibleEmbeddingRequestForDashScope() {
        EnterpriseRagJavaBenchmark.Config config = EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                "--questions", "questions.json",
                "--output", "summary.json",
                "--details-output", "details.jsonl",
                "--embedding-model", "text-embedding-v4",
                "--embedding-dimension", "2048",
                "--embedding-api-format", "openai",
                "--embedding-api-key", "test-token"
        });

        JsonNode body = EnterpriseRagJavaBenchmark.embeddingRequestBody(config, "How long is access valid?");

        assertThat(config.embeddingApiKey()).isEqualTo("test-token");
        assertThat(body.path("dimensions").asInt()).isEqualTo(2048);
        assertThat(body.has("dimension")).isFalse();
        assertThat(body.has("input_type")).isFalse();
    }

    @Test
    void buildsNativeDashScopeQueryRequestWithSeparateInstruction() {
        EnterpriseRagJavaBenchmark.Config config = EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                "--questions", "questions.json",
                "--output", "summary.json",
                "--details-output", "details.jsonl",
                "--embedding-model", "text-embedding-v4",
                "--embedding-dimension", "2048",
                "--embedding-api-format", "dashscope",
                "--embedding-query-instruction", "Retrieve enterprise passages"
        });

        JsonNode body = EnterpriseRagJavaBenchmark.embeddingRequestBody(config, "How long is access valid?");

        assertThat(body.path("input").path("texts").path(0).asText())
                .isEqualTo("How long is access valid?");
        assertThat(body.path("parameters").path("text_type").asText()).isEqualTo("query");
        assertThat(body.path("parameters").path("dimension").asInt()).isEqualTo(2048);
        assertThat(body.path("parameters").path("output_type").asText()).isEqualTo("dense");
        assertThat(body.path("parameters").path("instruct").asText())
                .isEqualTo("Retrieve enterprise passages");
        assertThat(body.has("encoding_format")).isFalse();
    }

    @Test
    void buildsEnglishBm25QueryAgainstStemmedSubfields() {
        EnterpriseRagJavaBenchmark.Config config = EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                "--questions", "questions.json",
                "--output", "summary.json",
                "--details-output", "details.jsonl",
                "--keyword-bm25-enabled", "true",
                "--english-bm25-enabled", "true"
        });

        JsonNode body = EnterpriseRagJavaBenchmark.bm25SearchBody(
                config,
                "contractor access expires",
                EnterpriseRagJavaBenchmark.sourceAclFilter(List.of("confluence")),
                List.of("title.english^2.0", "textContent.english^1.0"));

        assertThat(config.keywordBm25Enabled()).isTrue();
        assertThat(config.keywordBm25Weight()).isEqualTo(1.25d);
        assertThat(config.englishBm25Enabled()).isTrue();
        assertThat(config.englishBm25Weight()).isEqualTo(1.5d);
        assertThat(body.toString()).contains("title.english^2.0");
        assertThat(body.toString()).contains("textContent.english^1.0");
        assertThat(body.toString()).contains("source:confluence");
    }

    @Test
    void buildsBoundedEvidenceQueryWithTheSameAclScope() {
        EnterpriseRagJavaBenchmark.Config config = EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                "--questions", "questions.json",
                "--output", "summary.json",
                "--details-output", "details.jsonl",
                "--keyword-bm25-enabled", "true",
                "--english-bm25-enabled", "true",
                "--evidence-enabled", "true",
                "--evidence-candidate-chunks-per-document", "6"
        });

        JsonNode body = EnterpriseRagJavaBenchmark.evidenceSearchBody(
                config,
                "What are the multipart upload limits?",
                EnterpriseRagJavaBenchmark.sourceAclFilter(List.of("github")),
                List.of("doc-1", "doc-2"));
        String json = body.toString();

        assertThat(body.path("size").asInt()).isEqualTo(2);
        assertThat(body.path("collapse").path("field").asText()).isEqualTo("benchmarkDocId");
        assertThat(body.path("collapse").path("inner_hits").path("size").asInt()).isEqualTo(6);
        assertThat(json).contains("doc-1", "doc-2", "source:github");
        assertThat(json).contains("title.english^2.0", "textContent.english^1.0");
        assertThat(json).contains("multi_match").doesNotContain("question_type");
        assertThat(config.evidenceConfig().chunksPerDocument()).isEqualTo(3);
        assertThat(config.evidenceConfig().tokenBudget()).isEqualTo(6000);
    }

    @Test
    void keepsEvidenceDisabledByDefaultSoExistingRetrievalRunsRemainComparable() {
        EnterpriseRagJavaBenchmark.Config config = EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                "--questions", "questions.json",
                "--output", "summary.json",
                "--details-output", "details.jsonl"
        });

        assertThat(config.evidenceEnabled()).isFalse();
        assertThat(config.runId()).isEqualTo("paismart_java_enterpriserag_source_acl_hybrid");
        assertThat(config.manifestOutput().toString()).endsWith("summary.manifest.json");
        assertThat(config.evidenceOutput().toString()).endsWith("details.evidence.jsonl");
    }

    @Test
    void exportsEvidenceUsingTheExistingPythonContextsContract() {
        EnterpriseRagJavaBenchmark.Config config = EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                "--questions", "questions.json",
                "--output", "summary.json",
                "--details-output", "details.jsonl",
                "--evidence-enabled", "true"
        });
        EvidenceBuilder.RouteSignal signal = new EvidenceBuilder.RouteSignal(
                "bm25_original", 1, 1.0d, 9.0d, 1.0d / 11.0d, "doc-1:00002");
        EnterpriseRagJavaBenchmark.RankedDocument document = new EnterpriseRagJavaBenchmark.RankedDocument(
                "doc-1",
                "doc-1:00002",
                2,
                "github",
                "github:upload-limits",
                "Upload limits",
                "The total request limit is 50 MiB.",
                "internal",
                "v2",
                "document-hash",
                "2026-08-17T00:00:00Z",
                "chunk-hash",
                "2026-08-18T00:00:00Z",
                0.9d);
        EvidenceBuilder.EvidenceSpan span = new EvidenceBuilder.EvidenceSpan(
                "S1",
                1,
                "doc-1",
                1,
                "doc-1:00002",
                2,
                "github",
                "github:upload-limits",
                "Upload limits",
                "internal",
                "v2",
                "document-hash",
                "2026-08-17T00:00:00Z",
                "chunk-hash",
                "The total request limit is 50 MiB.",
                7,
                0.9d,
                0.8d,
                1.0d,
                9.0d,
                1.0d,
                1.0d,
                List.of(signal),
                "");
        EnterpriseRagJavaBenchmark.Result result = new EnterpriseRagJavaBenchmark.Result(
                List.of(document),
                Map.of("doc-1", List.of(new EnterpriseRagJavaBenchmark.RouteEvidence(signal, document))),
                new EvidenceBuilder.EvidenceBundle(List.of(span), 7, List.of(), 1, 2),
                new EnterpriseRagJavaBenchmark.EvidenceScores(1.0d, 1.0d, 1.0d),
                1.0d,
                2.0d,
                3.0d,
                0.0d,
                0.0d,
                0.1d,
                4.0d,
                6.1d,
                10.1d);
        EnterpriseRagJavaBenchmark.Question question = new EnterpriseRagJavaBenchmark.Question(
                "q1",
                "What is the total request limit?",
                List.of("doc-1"),
                List.of("github"),
                "basic",
                "50 MiB",
                List.of("The total request limit is 50 MiB."));

        JsonNode row = MAPPER.valueToTree(result.toEvidenceRow(config, question));

        assertThat(row.path("qid").asText()).isEqualTo("q1");
        assertThat(row.path("context_mode").asText()).isEqualTo("java_evidence_builder");
        assertThat(row.path("retrieval_hit_at_10").asBoolean()).isTrue();
        assertThat(row.path("contexts").path(0).path("citation_id").asText()).isEqualTo("S1");
        assertThat(row.path("contexts").path(0).path("source_type").asText()).isEqualTo("github");
        assertThat(row.path("contexts").path(0).path("text").asText()).contains("50 MiB");
        assertThat(row.path("contexts").path(0).path("document_version").asText()).isEqualTo("v2");
        assertThat(row.path("contexts").path(0).path("route_signals").path(0).path("route").asText())
                .isEqualTo("bm25_original");
    }

    @Test
    void excludesNoGoldQuestionsFromEvidenceAnswerMetrics() {
        EnterpriseRagJavaBenchmark.Question noGold = new EnterpriseRagJavaBenchmark.Question(
                "q-no-gold",
                "What is the unsupported retention period?",
                List.of(),
                List.of(),
                "info_not_found",
                "The corpus does not contain this information.",
                List.of("The retention period is unavailable."));

        EnterpriseRagJavaBenchmark.EvidenceScores scores = EnterpriseRagJavaBenchmark.scoreEvidence(
                noGold,
                new EvidenceBuilder.EvidenceBundle(List.of(), 0, List.of(), 0, 0));

        assertThat(scores.factTokenRecall()).isNull();
        assertThat(scores.factCoverage()).isNull();
        assertThat(scores.goldAnswerTokenRecall()).isNull();
    }

    @Test
    void rejectsStaleEmbeddingDimensionMetadataInExperimentConfig(@TempDir Path tempDir) throws Exception {
        Path configFile = tempDir.resolve("experiment.json");
        Files.writeString(configFile, """
                {
                  "schema_version": 1,
                  "name": "dimension-mismatch",
                  "arguments": {
                    "questions": "questions.json",
                    "output": "summary.json",
                    "details-output": "details.jsonl",
                    "embedding-model": "Qwen/Qwen3-Embedding-4B",
                    "embedding-dimension": 2560
                  },
                  "metadata": {
                    "embedding": {
                      "model": "Qwen/Qwen3-Embedding-4B",
                      "native_dimension": 2560,
                      "stored_dimension": 2048
                    }
                  }
                }
                """);

        assertThatThrownBy(() -> EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                "--config", configFile.toString()
        }))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("stored_dimension")
                .hasMessageContaining("2048 != 2560");
    }

    @Test
    void validatesLiveIndexDimensionAndEvidenceFieldsBeforeRunning() throws Exception {
        EnterpriseRagJavaBenchmark.Config config = EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                "--questions", "questions.json",
                "--output", "summary.json",
                "--details-output", "details.jsonl",
                "--embedding-dimension", "2048",
                "--evidence-enabled", "true"
        });
        JsonNode validMappings = MAPPER.readTree("""
                {
                  "_meta": {
                    "embedding_model": "intfloat/multilingual-e5-small",
                    "embedding_dimension": 2048
                  },
                  "properties": {
                    "vector": {"type": "dense_vector", "dims": 2048},
                    "documentVersion": {"type": "keyword"},
                    "documentHash": {"type": "keyword"},
                    "contentHash": {"type": "keyword"},
                    "sourceUpdatedAt": {"type": "date"}
                  }
                }
                """);

        EnterpriseRagJavaBenchmark.validateIndexMetadata(config, validMappings);

        JsonNode wrongDimension = validMappings.deepCopy();
        ((com.fasterxml.jackson.databind.node.ObjectNode) wrongDimension.path("properties").path("vector"))
                .put("dims", 2560);
        assertThatThrownBy(() -> EnterpriseRagJavaBenchmark.validateIndexMetadata(config, wrongDimension))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("config=2048")
                .hasMessageContaining("mapping=2560");

        JsonNode wrongModel = validMappings.deepCopy();
        ((com.fasterxml.jackson.databind.node.ObjectNode) wrongModel.path("_meta"))
                .put("embedding_model", "Qwen/Qwen3-Embedding-4B");
        assertThatThrownBy(() -> EnterpriseRagJavaBenchmark.validateIndexMetadata(config, wrongModel))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("embedding model mismatch")
                .hasMessageContaining("multilingual-e5-small")
                .hasMessageContaining("Qwen/Qwen3-Embedding-4B");

        JsonNode missingVersionField = validMappings.deepCopy();
        ((com.fasterxml.jackson.databind.node.ObjectNode) missingVersionField.path("properties"))
                .remove("documentHash");
        assertThatThrownBy(() -> EnterpriseRagJavaBenchmark.validateIndexMetadata(config, missingVersionField))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("documentHash");
    }

    @Test
    void buildsOpenSearchKnnQueryInsideQueryDsl() {
        EnterpriseRagJavaBenchmark.Config config = EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                "--questions", "questions.json",
                "--output", "summary.json",
                "--details-output", "details.jsonl",
                "--engine", "opensearch"
        });

        JsonNode body = EnterpriseRagJavaBenchmark.denseSearchBody(
                config,
                List.of(0.1d, 0.2d),
                EnterpriseRagJavaBenchmark.sourceAclFilter(List.of("jira")));

        JsonNode vectorQuery = body.path("query").path("knn").path("vector");
        assertThat(vectorQuery.path("vector").isArray()).isTrue();
        assertThat(vectorQuery.path("k").asInt()).isEqualTo(500);
        assertThat(vectorQuery.path("method_parameters").path("ef_search").asInt()).isEqualTo(2500);
        assertThat(vectorQuery.path("filter").toString()).contains("source:jira");
        assertThat(body.has("knn")).isFalse();
    }
}
