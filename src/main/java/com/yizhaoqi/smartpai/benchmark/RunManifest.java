package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/** Writes an auditable manifest before, during, and after an experiment run. */
final class RunManifest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int SCHEMA_VERSION = 1;

    private RunManifest() {
    }

    static Session start(
            Path output,
            String runId,
            ExperimentConfig.Snapshot experiment,
            Path questions,
            Map<String, Object> execution) throws IOException {
        ObjectNode root = MAPPER.createObjectNode();
        root.put("schema_version", SCHEMA_VERSION);
        root.put("run_id", runId);
        root.put("status", "running");
        root.put("started_at", now());

        ObjectNode experimentNode = root.putObject("experiment");
        experimentNode.put("name", experiment.name());
        experimentNode.put(
                "config_path",
                experiment.configPath() == null ? "<inline-cli>" : experiment.configPath().toString());
        experimentNode.put("config_sha256", experiment.configSha256());
        experimentNode.put("base_dir", experiment.baseDir().toString());
        experimentNode.set("metadata", experiment.metadata());
        experimentNode.set("resolved_arguments", MAPPER.valueToTree(experiment.sanitizedArguments()));
        experimentNode.put(
                "resolved_arguments_sha256",
                sha256(MAPPER.writeValueAsBytes(experiment.sanitizedArguments())));

        root.set("code", codeVersion(experiment.baseDir()));
        root.set("runtime", runtime());
        Map<String, Object> canonicalExecution = new TreeMap<>(execution);
        root.set("execution", MAPPER.valueToTree(canonicalExecution));
        root.put("execution_sha256", sha256(MAPPER.writeValueAsBytes(canonicalExecution)));

        ObjectNode inputs = root.putObject("inputs");
        experiment.inputsWith("questions", questions).forEach((name, path) ->
                inputs.set(name, fingerprint(path)));

        Session session = new Session(output.toAbsolutePath().normalize(), root);
        session.write();
        return session;
    }

    private static ObjectNode runtime() {
        ObjectNode runtime = MAPPER.createObjectNode();
        runtime.put("java_version", System.getProperty("java.version", "unknown"));
        runtime.put("java_vendor", System.getProperty("java.vendor", "unknown"));
        runtime.put("java_vm", System.getProperty("java.vm.name", "unknown"));
        runtime.put("os_name", System.getProperty("os.name", "unknown"));
        runtime.put("os_version", System.getProperty("os.version", "unknown"));
        runtime.put("os_arch", System.getProperty("os.arch", "unknown"));
        runtime.put("available_processors", Runtime.getRuntime().availableProcessors());
        runtime.put("cwd", Path.of("").toAbsolutePath().normalize().toString());
        return runtime;
    }

    private static ObjectNode codeVersion(Path baseDir) {
        ObjectNode code = MAPPER.createObjectNode();
        CommandResult commit = command(baseDir, "git", "rev-parse", "HEAD");
        CommandResult branch = command(baseDir, "git", "branch", "--show-current");
        CommandResult status = command(baseDir, "git", "status", "--porcelain");
        code.put("git_root", baseDir.toString());
        code.put("commit", commit.success() ? commit.output().trim() : "unknown");
        code.put("branch", branch.success() ? branch.output().trim() : "unknown");
        boolean dirty = status.success() && !status.output().isBlank();
        code.put("dirty", dirty);
        code.put("status", status.success() ? status.output().trim() : "unknown");
        if (dirty) {
            CommandResult diff = command(baseDir, "git", "diff", "--binary", "HEAD");
            code.put(
                    "tracked_diff_sha256",
                    diff.success()
                            ? sha256(diff.output().getBytes(StandardCharsets.UTF_8))
                            : "unavailable");
        }
        if (!commit.success()) {
            code.put("git_error", commit.output().trim());
        }
        return code;
    }

    private static CommandResult command(Path directory, String... command) {
        Process process = null;
        Thread reader = null;
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        AtomicReference<IOException> readFailure = new AtomicReference<>();
        try {
            process = new ProcessBuilder(command)
                    .directory(directory.toFile())
                    .redirectErrorStream(true)
                    .start();
            Process runningProcess = process;
            reader = new Thread(() -> {
                try (var input = runningProcess.getInputStream()) {
                    input.transferTo(output);
                } catch (IOException exception) {
                    readFailure.set(exception);
                }
            }, "run-manifest-command-reader");
            reader.setDaemon(true);
            reader.start();

            boolean completed = process.waitFor(5, TimeUnit.SECONDS);
            if (!completed) {
                process.destroyForcibly();
                process.waitFor(1, TimeUnit.SECONDS);
            }
            reader.join(1_000L);
            if (reader.isAlive()) {
                reader.interrupt();
                return new CommandResult(false, "command output reader timed out");
            }
            if (!completed) {
                return new CommandResult(false, "command timed out");
            }
            if (readFailure.get() != null) {
                return new CommandResult(false, "command output read failed: " + readFailure.get().getMessage());
            }
            return new CommandResult(
                    process.exitValue() == 0,
                    output.toString(StandardCharsets.UTF_8));
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            if (process != null) {
                process.destroyForcibly();
            }
            return new CommandResult(false, "command interrupted");
        } catch (Exception exception) {
            if (process != null) {
                process.destroyForcibly();
            }
            return new CommandResult(false, exception.getClass().getSimpleName() + ": " + exception.getMessage());
        }
    }

    private static ObjectNode fingerprint(Path input) {
        ObjectNode node = MAPPER.createObjectNode();
        Path normalized = input.toAbsolutePath().normalize();
        node.put("path", normalized.toString());
        boolean exists = Files.isRegularFile(normalized);
        node.put("exists", exists);
        if (!exists) {
            return node;
        }
        try {
            node.put("size_bytes", Files.size(normalized));
            node.put("modified_at", Files.getLastModifiedTime(normalized).toInstant().toString());
            node.put("sha256", sha256(normalized));
        } catch (IOException exception) {
            node.put("fingerprint_error", exception.getMessage());
        }
        return node;
    }

    private static String sha256(Path path) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (var input = Files.newInputStream(path)) {
                byte[] buffer = new byte[64 * 1024];
                int read;
                while ((read = input.read(buffer)) >= 0) {
                    if (read > 0) {
                        digest.update(buffer, 0, read);
                    }
                }
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is unavailable", exception);
        }
    }

    static String sha256Json(JsonNode node) throws IOException {
        return sha256(MAPPER.writeValueAsBytes(node));
    }

    private static String sha256(byte[] bytes) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is unavailable", exception);
        }
    }

    private static String now() {
        return OffsetDateTime.now(ZoneOffset.UTC).toString();
    }

    static final class Session {
        private final Path output;
        private final ObjectNode root;

        private Session(Path output, ObjectNode root) {
            this.output = output;
            this.root = root;
        }

        void setIndexMetadata(JsonNode metadata) throws IOException {
            root.set("index_metadata", metadata == null ? MAPPER.createObjectNode() : metadata);
            write();
        }

        void complete(Map<String, Object> summary) throws IOException {
            root.put("status", "completed");
            root.put("completed_at", now());
            root.set("summary", MAPPER.valueToTree(summary));
            write();
        }

        void fail(Throwable error) {
            try {
                root.put("status", "failed");
                root.put("completed_at", now());
                ObjectNode failure = root.putObject("failure");
                failure.put("type", error.getClass().getName());
                failure.put("message", String.valueOf(error.getMessage()));
                write();
            } catch (IOException manifestFailure) {
                error.addSuppressed(manifestFailure);
            }
        }

        Path output() {
            return output;
        }

        private void write() throws IOException {
            Path parent = output.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Path temporary = output.resolveSibling(output.getFileName() + ".tmp");
            MAPPER.writerWithDefaultPrettyPrinter().writeValue(temporary.toFile(), root);
            try {
                Files.move(
                        temporary,
                        output,
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException ignored) {
                Files.move(temporary, output, StandardCopyOption.REPLACE_EXISTING);
            }
        }
    }

    private record CommandResult(boolean success, String output) {
    }
}
