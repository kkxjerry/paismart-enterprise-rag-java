package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
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
        Path details = tempDir.resolve("details.jsonl");
        Files.writeString(details, "{\"qid\":\"q1\"}\n");
        session.complete(Map.of("hit@10", 0.98d), Map.of("details", details));
        JsonNode completed = MAPPER.readTree(manifestPath.toFile());

        assertThat(completed.path("status").asText()).isEqualTo("completed");
        assertThat(completed.path("completed_at").asText()).isNotBlank();
        assertThat(completed.path("index_metadata").path("vector_dimension").asInt()).isEqualTo(2048);
        assertThat(completed.path("summary").path("hit@10").asDouble()).isEqualTo(0.98d);
        assertThat(completed.path("outputs").path("details").path("sha256").asText()).hasSize(64);
        assertThat(completed.path("outputs").path("details").path("size_bytes").asLong())
                .isEqualTo(Files.size(details));
        assertThat(Files.exists(manifestPath.resolveSibling("run.manifest.json.tmp"))).isFalse();
    }

    @Test
    void fingerprintsUntrackedFilesInADirtyGitCheckout() throws Exception {
        Path repo = tempDir.resolve("repo");
        Files.createDirectories(repo);
        Files.writeString(repo.resolve("questions.json"), "[]\n");
        Path config = repo.resolve("experiment.json");
        Files.writeString(config, """
                {
                  "schema_version": 1,
                  "name": "git-fingerprint",
                  "base_dir": ".",
                  "arguments": {
                    "questions": "questions.json",
                    "output": "summary.json",
                    "details-output": "details.jsonl"
                  }
                }
                """);
        git(repo, "init", "-q");
        git(repo, "config", "user.email", "test@example.com");
        git(repo, "config", "user.name", "Test User");
        git(repo, "add", ".");
        git(repo, "commit", "-qm", "initial");
        Path untracked = repo.resolve("UntrackedEvidence.java");
        Files.writeString(untracked, "final class UntrackedEvidence {}\n");

        ExperimentConfig.Snapshot experiment = ExperimentConfig.load(new String[] {
                "--config", config.toString()
        });
        Path manifestPath = repo.resolve("manifest.json");
        RunManifest.start(
                manifestPath,
                "git-fingerprint",
                experiment,
                repo.resolve("questions.json"),
                Map.of());
        JsonNode manifest = MAPPER.readTree(manifestPath.toFile());

        assertThat(manifest.path("code").path("dirty").asBoolean()).isTrue();
        assertThat(manifest.path("code").path("untracked_file_count").asInt()).isEqualTo(1);
        assertThat(manifest.path("code").path("untracked_files")
                .path("UntrackedEvidence.java").path("sha256").asText()).hasSize(64);
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

    private static void git(Path directory, String... arguments) throws Exception {
        String[] command = new String[arguments.length + 1];
        command[0] = "git";
        System.arraycopy(arguments, 0, command, 1, arguments.length);
        Process process = new ProcessBuilder(command)
                .directory(directory.toFile())
                .redirectErrorStream(true)
                .start();
        String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        assertThat(process.waitFor())
                .withFailMessage("git command failed: %s", output)
                .isZero();
    }
}
