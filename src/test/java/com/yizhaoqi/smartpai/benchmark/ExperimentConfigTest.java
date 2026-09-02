package com.yizhaoqi.smartpai.benchmark;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ExperimentConfigTest {

    @TempDir
    Path tempDir;

    @Test
    void resolvesPathsAndLetsExplicitCliArgumentsOverrideTheConfig() throws Exception {
        Path project = tempDir.resolve("project");
        Path configDir = project.resolve("config/experiments");
        Files.createDirectories(configDir);
        Files.createDirectories(project.resolve("data"));
        Files.writeString(project.resolve("data/questions.json"), "[]");
        Files.writeString(project.resolve("data/questions-override.json"), "[]");
        Files.writeString(project.resolve("data/docs.jsonl"), "{}\n");
        Path config = configDir.resolve("evidence.json");
        Files.writeString(config, """
                {
                  "schema_version": 1,
                  "name": "sample-evidence",
                  "base_dir": "../..",
                  "arguments": {
                    "questions": "data/questions.json",
                    "output": "runs/summary.json",
                    "details-output": "runs/details.jsonl",
                    "top-k": 50,
                    "evidence-enabled": true
                  },
                  "inputs": {
                    "questions": "data/questions.json",
                    "documents": "data/docs.jsonl"
                  },
                  "metadata": {
                    "embedding": {
                      "native_dimension": 2560,
                      "stored_dimension": 2048
                    }
                  }
                }
                """);

        Path overriddenQuestions = project.resolve("data/questions-override.json");
        ExperimentConfig.Snapshot snapshot = ExperimentConfig.load(new String[] {
                "--config", config.toString(),
                "--questions", overriddenQuestions.toString(),
                "--top-k", "7"
        });

        assertThat(snapshot.name()).isEqualTo("sample-evidence");
        assertThat(snapshot.baseDir()).isEqualTo(project.toAbsolutePath().normalize());
        assertThat(snapshot.arguments().requiredPath("questions"))
                .isEqualTo(overriddenQuestions.toAbsolutePath().normalize());
        assertThat(snapshot.inputsWith("questions", snapshot.arguments().requiredPath("questions")).get("questions"))
                .isEqualTo(overriddenQuestions.toAbsolutePath().normalize());
        assertThat(snapshot.arguments().requiredPath("output"))
                .isEqualTo(project.resolve("runs/summary.json").toAbsolutePath().normalize());
        assertThat(snapshot.arguments().positiveInt("top-k", 50)).isEqualTo(7);
        assertThat(snapshot.inputs().get("documents"))
                .isEqualTo(project.resolve("data/docs.jsonl").toAbsolutePath().normalize());
        assertThat(snapshot.metadata().path("embedding").path("native_dimension").asInt())
                .isEqualTo(2560);
        assertThat(snapshot.configSha256()).hasSize(64);
    }

    @Test
    void allCheckedInExperimentConfigsResolveToValidEffectiveEvaluateConfigs() throws Exception {
        try (var paths = Files.list(Path.of("config/experiments"))) {
            List<Path> configs = paths
                    .filter(path -> path.getFileName().toString().endsWith(".json"))
                    .sorted()
                    .toList();
            assertThat(configs).isNotEmpty();
            for (Path configPath : configs) {
                EnterpriseRagJavaBenchmark.Config config = EnterpriseRagJavaBenchmark.Config.parse(new String[] {
                        "--config", configPath.toString()
                });
                assertThat(config.runId()).isNotBlank();
                assertThat(config.questions()).isAbsolute();
                assertThat(config.output()).isAbsolute();
                assertThat(config.manifestOutput()).isAbsolute();
            }
        }
    }

    @Test
    void rejectsSecretsStoredInsideTheVersionedConfig() throws Exception {
        Path config = tempDir.resolve("bad-config.json");
        Files.writeString(config, """
                {
                  "schema_version": 1,
                  "name": "bad",
                  "arguments": {
                    "questions": "questions.json",
                    "output": "summary.json",
                    "details-output": "details.jsonl",
                    "embedding-api-key": "must-not-be-versioned"
                  }
                }
                """);

        assertThatThrownBy(() -> ExperimentConfig.load(new String[] {
                "--config", config.toString()
        }))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("do not store secrets");
    }

    @Test
    void rejectsSecretsHiddenInsideMetadata() throws Exception {
        Path config = tempDir.resolve("bad-metadata-config.json");
        Files.writeString(config, """
                {
                  "schema_version": 1,
                  "name": "bad-metadata",
                  "arguments": {
                    "questions": "questions.json",
                    "output": "summary.json",
                    "details-output": "details.jsonl"
                  },
                  "metadata": {
                    "runtime": {
                      "dashscope_api_key": "must-not-be-versioned"
                    }
                  }
                }
                """);

        assertThatThrownBy(() -> ExperimentConfig.load(new String[] {
                "--config", config.toString()
        }))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("experiment config metadata")
                .hasMessageContaining("dashscope_api_key");
    }

    @Test
    void redactsCliSecretsButKeepsNonSecretTokenSettings() throws Exception {
        ExperimentConfig.Snapshot snapshot = ExperimentConfig.load(new String[] {
                "--questions", "questions.json",
                "--output", "summary.json",
                "--details-output", "details.jsonl",
                "--embedding-api-key", "top-secret",
                "--embedding-api-key-env", "MY_EMBEDDING_KEY",
                "--evidence-token-budget", "6000"
        });

        assertThat(snapshot.sanitizedArguments().get("embedding-api-key")).isEqualTo("<redacted>");
        assertThat(snapshot.sanitizedArguments().get("embedding-api-key-env")).isEqualTo("MY_EMBEDDING_KEY");
        assertThat(snapshot.sanitizedArguments().get("evidence-token-budget")).isEqualTo("6000");
    }
}
