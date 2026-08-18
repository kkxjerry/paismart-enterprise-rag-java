package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class EnterpriseRagJavaBenchmarkTest {

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
