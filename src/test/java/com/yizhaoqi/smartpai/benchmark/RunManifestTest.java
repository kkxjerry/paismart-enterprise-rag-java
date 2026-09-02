package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class RunManifestTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @TempDir
    Path tempDir;

    @Test
    void recordsInputFingerprintsRedactsSecretsAndCompletesAtomically() throws Exception {
        Path questions = tempDir.resolve("questions.json");
        Files.writeString(questions, "[{\"question\":\"hello\"}]\n");
        ExperimentConfig.Snapshot experiment = ExperimentConfig.load(new String[] {
                "--questions", questions.toString(),
                "--output", tempDir.resolve("summary.json").toString(),
                "--details-output", tempDir.resolve("details.jsonl").toString(),
                "--embedding-api-key", "never-write-this"
        });
        Path manifestPath = tempDir.resolve("run.manifest.json");

        RunManifest.Session session = RunManifest.start(
                manifestPath,
                "run-test",
                experiment,
                questions,
                Map.of("evidence_enabled", true));
        JsonNode running = MAPPER.readTree(manifestPath.toFile());

        assertThat(running.path("status").asText()).isEqualTo("running");
        assertThat(running.path("inputs").path("questions").path("sha256").asText()).hasSize(64);
        assertThat(running.path("execution_sha256").asText()).hasSize(64);
        assertThat(running.path("experiment").path("resolved_arguments").path("embedding-api-key").asText())
                .isEqualTo("<redacted>");
        assertThat(Files.readString(manifestPath)).doesNotContain("never-write-this");

        session.setIndexMetadata(MAPPER.readTree("{\"vector_dimension\":2048}"));
        session.complete(Map.of("hit@10", 0.98d));
        JsonNode completed = MAPPER.readTree(manifestPath.toFile());

        assertThat(completed.path("status").asText()).isEqualTo("completed");
        assertThat(completed.path("completed_at").asText()).isNotBlank();
        assertThat(completed.path("index_metadata").path("vector_dimension").asInt()).isEqualTo(2048);
        assertThat(completed.path("summary").path("hit@10").asDouble()).isEqualTo(0.98d);
        assertThat(Files.exists(manifestPath.resolveSibling("run.manifest.json.tmp"))).isFalse();
    }

    @Test
    void persistsFailureState() throws Exception {
        Path questions = tempDir.resolve("questions-failure.json");
        Files.writeString(questions, "[]");
        ExperimentConfig.Snapshot experiment = ExperimentConfig.load(new String[] {
                "--questions", questions.toString(),
                "--output", tempDir.resolve("summary-failure.json").toString(),
                "--details-output", tempDir.resolve("details-failure.jsonl").toString()
        });
        Path manifestPath = tempDir.resolve("failure.manifest.json");
        RunManifest.Session session = RunManifest.start(
                manifestPath,
                "run-failure",
                experiment,
                questions,
                Map.of());

        session.fail(new IllegalStateException("index unavailable"));
        JsonNode failed = MAPPER.readTree(manifestPath.toFile());

        assertThat(failed.path("status").asText()).isEqualTo("failed");
        assertThat(failed.path("failure").path("type").asText())
                .isEqualTo(IllegalStateException.class.getName());
        assertThat(failed.path("failure").path("message").asText()).isEqualTo("index unavailable");
    }
}
