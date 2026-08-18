package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class ElasticsearchIndexCommandTest {

    @Test
    void buildsEnglishBm25AndDenseVectorMapping() {
        ElasticsearchIndexCommand.Config config = ElasticsearchIndexCommand.Config.parse(new String[] {
                "--index", "benchmark_v1",
                "--embedding-model", "Qwen/Qwen3-Embedding-4B",
                "--embedding-dimension", "2048"
        });

        JsonNode mapping = ElasticsearchIndexCommand.buildMapping(config);

        assertThat(mapping.path("settings").path("similarity").path("enterprise_bm25").path("k1").asDouble())
                .isEqualTo(2.2d);
        assertThat(mapping.path("mappings").path("properties").path("textContent").path("analyzer").asText())
                .isEqualTo("standard");
        assertThat(mapping.path("mappings").path("properties").path("textContent")
                .path("fields").path("english").path("analyzer").asText()).isEqualTo("english");
        assertThat(mapping.path("mappings").path("properties").path("vector").path("dims").asInt())
                .isEqualTo(2048);
    }
}
