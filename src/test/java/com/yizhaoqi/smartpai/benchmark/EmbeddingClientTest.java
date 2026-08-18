package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class EmbeddingClientTest {

    @Test
    void localRequestKeepsDocumentAndQueryRolesSeparate() {
        EmbeddingClient client = new EmbeddingClient(config("local", "", "Retrieve enterprise passages"));

        JsonNode document = client.requestBody(List.of("document"), EmbeddingClient.InputType.DOCUMENT);
        JsonNode query = client.requestBody(List.of("question"), EmbeddingClient.InputType.QUERY);

        assertThat(document.path("input_type").asText()).isEqualTo("passage");
        assertThat(document.path("input").path(0).asText()).isEqualTo("document");
        assertThat(query.path("input_type").asText()).isEqualTo("query");
        assertThat(query.path("input").path(0).asText())
                .isEqualTo("Instruct: Retrieve enterprise passages\nQuery:question");
    }

    @Test
    void nativeDashScopeUsesStructuredTextType() {
        EmbeddingClient client = new EmbeddingClient(config("dashscope", "token", "Retrieve passages"));

        JsonNode body = client.requestBody(List.of("question"), EmbeddingClient.InputType.QUERY);

        assertThat(body.path("input").path("texts").path(0).asText()).isEqualTo("question");
        assertThat(body.path("parameters").path("text_type").asText()).isEqualTo("query");
        assertThat(body.path("parameters").path("dimension").asInt()).isEqualTo(2048);
        assertThat(body.path("parameters").path("instruct").asText()).isEqualTo("Retrieve passages");
    }

    private static EmbeddingClient.Config config(String format, String key, String instruction) {
        return new EmbeddingClient.Config(
                "http://localhost/embeddings",
                "test-model",
                2048,
                format,
                key,
                instruction,
                3,
                Duration.ofSeconds(30));
    }
}
