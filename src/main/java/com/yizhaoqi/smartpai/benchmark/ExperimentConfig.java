package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Loads one versioned experiment file and merges explicit CLI overrides.
 *
 * <p>The JSON shape is intentionally small and shared by retrieval, evidence,
 * and downstream offline evaluation:</p>
 *
 * <pre>{@code
 * {
 *   "schema_version": 1,
 *   "name": "enterpriserag-qwen3-evidence-v1",
 *   "base_dir": "../..",
 *   "arguments": { "questions": "data/...", "evidence-enabled": true },
 *   "inputs": { "questions": "data/...", "documents": "data/..." },
 *   "metadata": { "embedding": { "native_dimension": 2560 } }
 * }
 * }</pre>
 */
final class ExperimentConfig {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int SCHEMA_VERSION = 1;
    private static final Set<String> PATH_ARGUMENTS = Set.of(
            "questions",
            "output",
            "details-output",
            "manifest-output",
            "evidence-output");
    private static final Set<String> SECRET_SUFFIXES = Set.of(
            "api-key",
            "bearer-token",
            "password",
            "secret");

    private ExperimentConfig() {
    }

    static Snapshot load(String[] rawArgs) throws IOException {
        Arguments cli = Arguments.parse(rawArgs);
        String configValue = cli.string("config", "").trim();
        if (configValue.isEmpty()) {
            Map<String, String> values = new LinkedHashMap<>(cli.asMap());
            values.remove("config");
            Path cwd = Path.of("").toAbsolutePath().normalize();
            return new Snapshot(
                    "inline-cli",
                    null,
                    cwd,
                    Arguments.fromMap(values),
                    Map.of(),
                    MAPPER.createObjectNode(),
                    "");
        }

        Path configPath = Path.of(configValue).toAbsolutePath().normalize();
        if (!Files.isRegularFile(configPath)) {
            throw new IllegalArgumentException("experiment config is not a readable file: " + configPath);
        }
        JsonNode root = MAPPER.readTree(configPath.toFile());
        if (!root.isObject()) {
            throw new IllegalArgumentException("experiment config must be a JSON object: " + configPath);
        }
        int version = root.path("schema_version").asInt(-1);
        if (version != SCHEMA_VERSION) {
            throw new IllegalArgumentException(
                    "unsupported experiment config schema_version " + version + "; expected " + SCHEMA_VERSION);
        }

        Path configDirectory = configPath.getParent() == null
                ? Path.of("").toAbsolutePath().normalize()
                : configPath.getParent();
        String baseDirValue = root.path("base_dir").asText("").trim();
        Path baseDir = baseDirValue.isEmpty()
                ? configDirectory
                : resolve(configDirectory, baseDirValue);

        Map<String, String> values = scalarMap(root.path("arguments"), "arguments");
        for (String key : values.keySet()) {
            if (isSecretKey(key)) {
                throw new IllegalArgumentException(
                        "do not store secrets in experiment config arguments; use an environment variable for " + key);
            }
        }
        for (String key : PATH_ARGUMENTS) {
            String value = values.get(key);
            if (value != null && !value.isBlank()) {
                values.put(key, resolve(baseDir, value).toString());
            }
        }

        Map<String, String> cliValues = new LinkedHashMap<>(cli.asMap());
        cliValues.remove("config");
        values.putAll(cliValues);

        Map<String, Path> inputs = pathMap(root.path("inputs"), baseDir);
        String name = root.path("name").asText("").trim();
        if (name.isEmpty()) {
            String filename = configPath.getFileName().toString();
            int dot = filename.lastIndexOf('.');
            name = dot > 0 ? filename.substring(0, dot) : filename;
        }
        JsonNode metadata = root.path("metadata");
        if (!metadata.isObject()) {
            metadata = MAPPER.createObjectNode();
        }
        rejectSecretValues(metadata, "metadata");
        return new Snapshot(
                name,
                configPath,
                baseDir,
                Arguments.fromMap(values),
                Map.copyOf(inputs),
                sanitize(metadata),
                sha256(configPath));
    }

    private static Map<String, String> scalarMap(JsonNode node, String field) {
        Map<String, String> values = new LinkedHashMap<>();
        if (node.isMissingNode() || node.isNull()) {
            return values;
        }
        if (!node.isObject()) {
            throw new IllegalArgumentException(field + " must be a JSON object");
        }
        node.fields().forEachRemaining(entry -> {
            JsonNode value = entry.getValue();
            if (!value.isValueNode() || value.isNull()) {
                throw new IllegalArgumentException(
                        field + "." + entry.getKey() + " must be a string, number, or boolean");
            }
            values.put(entry.getKey(), value.asText());
        });
        return values;
    }

    private static Map<String, Path> pathMap(JsonNode node, Path baseDir) {
        Map<String, Path> values = new LinkedHashMap<>();
        if (node.isMissingNode() || node.isNull()) {
            return values;
        }
        if (!node.isObject()) {
            throw new IllegalArgumentException("inputs must be a JSON object");
        }
        node.fields().forEachRemaining(entry -> {
            if (!entry.getValue().isTextual() || entry.getValue().asText().isBlank()) {
                throw new IllegalArgumentException("inputs." + entry.getKey() + " must be a non-empty path string");
            }
            values.put(entry.getKey(), resolve(baseDir, entry.getValue().asText()));
        });
        return values;
    }

    private static Path resolve(Path baseDir, String value) {
        Path path = Path.of(value);
        return (path.isAbsolute() ? path : baseDir.resolve(path)).toAbsolutePath().normalize();
    }

    private static boolean isSecretKey(String key) {
        String normalized = key.toLowerCase().replace('_', '-');
        if (normalized.equals("authorization")) {
            return true;
        }
        return SECRET_SUFFIXES.stream().anyMatch(secret ->
                normalized.equals(secret) || normalized.endsWith("-" + secret));
    }

    private static void rejectSecretValues(JsonNode node, String path) {
        if (node.isObject()) {
            node.fields().forEachRemaining(entry -> {
                String childPath = path + "." + entry.getKey();
                if (isSecretKey(entry.getKey())
                        && !entry.getValue().isNull()
                        && !entry.getValue().asText("").isBlank()) {
                    throw new IllegalArgumentException(
                            "do not store secrets in experiment config metadata: " + childPath);
                }
                rejectSecretValues(entry.getValue(), childPath);
            });
        } else if (node.isArray()) {
            for (int index = 0; index < node.size(); index++) {
                rejectSecretValues(node.get(index), path + "[" + index + "]");
            }
        }
    }

    private static JsonNode sanitize(JsonNode node) {
        if (node.isObject()) {
            ObjectNode out = MAPPER.createObjectNode();
            node.fields().forEachRemaining(entry -> {
                if (isSecretKey(entry.getKey())) {
                    out.put(entry.getKey(), "<redacted>");
                } else {
                    out.set(entry.getKey(), sanitize(entry.getValue()));
                }
            });
            return out;
        }
        if (node.isArray()) {
            var out = MAPPER.createArrayNode();
            node.forEach(value -> out.add(sanitize(value)));
            return out;
        }
        return node.deepCopy();
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

    record Snapshot(
            String name,
            Path configPath,
            Path baseDir,
            Arguments arguments,
            Map<String, Path> inputs,
            JsonNode metadata,
            String configSha256) {

        Map<String, String> sanitizedArguments() {
            Map<String, String> values = new TreeMap<>();
            arguments.asMap().forEach((key, value) -> values.put(
                    key,
                    isSecretKey(key) ? "<redacted>" : value));
            return java.util.Collections.unmodifiableMap(values);
        }

        Map<String, Path> inputsWith(String name, Path path) {
            Map<String, Path> values = new LinkedHashMap<>(inputs);
            values.put(name, path.toAbsolutePath().normalize());
            return Map.copyOf(values);
        }
    }
}
