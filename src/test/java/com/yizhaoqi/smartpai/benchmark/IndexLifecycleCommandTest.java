package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class IndexLifecycleCommandTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void promotionAtomicallyRemovesOldIndicesAndAddsOneWriteIndex() {
        JsonNode body = IndexLifecycleCommand.promotionBody(
                "enterprise-rag-read",
                "rag-v3",
                List.of("rag-v1", "rag-v2"));

        assertThat(body.path("actions")).hasSize(3);
        assertThat(body.toString()).contains("rag-v1", "rag-v2", "rag-v3", "enterprise-rag-read");
        assertThat(body.toString()).contains("is_write_index");
    }

    @Test
    void promotionDoesNotRemoveTheTargetIfItAlreadyHasTheAlias() {
        JsonNode body = IndexLifecycleCommand.promotionBody(
                "enterprise-rag-read",
                "rag-v3",
                List.of("rag-v2", "rag-v3"));

        assertThat(body.path("actions")).hasSize(2);
        assertThat(body.path("actions").path(0).path("remove").path("index").asText())
                .isEqualTo("rag-v2");
        assertThat(body.path("actions").path(1).path("add").path("index").asText())
                .isEqualTo("rag-v3");
    }

    @Test
    void readinessRejectsRedEmptyOrTimedOutIndices() throws Exception {
        JsonNode ready = IndexLifecycleCommand.validateTargetReadiness(
                MAPPER.readTree("{\"status\":\"yellow\",\"timed_out\":false,\"active_primary_shards\":1}"),
                MAPPER.readTree("{\"count\":42}"),
                1);
        assertThat(ready.path("documents").asLong()).isEqualTo(42L);

        assertThatThrownBy(() -> IndexLifecycleCommand.validateTargetReadiness(
                MAPPER.readTree("{\"status\":\"red\",\"timed_out\":false,\"active_primary_shards\":0}"),
                MAPPER.readTree("{\"count\":42}"),
                1))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("not ready");
        assertThatThrownBy(() -> IndexLifecycleCommand.validateTargetReadiness(
                MAPPER.readTree("{\"status\":\"green\",\"timed_out\":false,\"active_primary_shards\":1}"),
                MAPPER.readTree("{\"count\":0}"),
                1))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("below minimum");
    }

    @Test
    void configRequiresAliasAndRestrictsActions() {
        assertThatThrownBy(() -> IndexLifecycleCommand.Config.parse(new String[] {
                "--action", "delete", "--alias", "rag"
        }))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("action");
        assertThatThrownBy(() -> IndexLifecycleCommand.Config.parse(new String[] {"--action", "status"}))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("alias");
    }
}
