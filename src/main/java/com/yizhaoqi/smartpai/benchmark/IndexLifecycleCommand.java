package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/** Atomic alias inspection/promotion for blue-green RAG indices. */
public final class IndexLifecycleCommand {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private IndexLifecycleCommand() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        JsonHttpClient http = new JsonHttpClient();
        switch (config.action()) {
            case "status" -> System.out.println(
                    MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(status(http, config)));
            case "promote" -> System.out.println(
                    MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(promote(http, config)));
            default -> throw new IllegalArgumentException("unsupported lifecycle action: " + config.action());
        }
    }

    static ObjectNode status(JsonHttpClient http, Config config) throws Exception {
        List<String> indices = aliasIndices(http, config.esUrl(), config.alias());
        ObjectNode row = MAPPER.createObjectNode();
        row.put("action", "status");
        row.put("alias", config.alias());
        row.set("indices", MAPPER.valueToTree(indices));
        row.put("count", indices.size());
        return row;
    }

    static ObjectNode promote(JsonHttpClient http, Config config) throws Exception {
        if (config.index().isBlank()) {
            throw new IllegalArgumentException("--index is required for promote");
        }
        String base = stripTrailingSlash(config.esUrl());
        JsonHttpClient.Response target = http.send(
                "HEAD",
                base + "/" + config.index(),
                null,
                "application/json",
                "",
                1,
                Duration.ofSeconds(15));
        if (target.status() != 200) {
            throw new IllegalStateException("target index does not exist: " + config.index());
        }
        ObjectNode readiness = targetReadiness(http, config);
        List<String> current = aliasIndices(http, config.esUrl(), config.alias());
        if (!config.expectedCurrentIndex().isBlank()
                && !current.equals(List.of(config.expectedCurrentIndex()))) {
            throw new IllegalStateException(
                    "alias compare-and-set failed: expected " + config.expectedCurrentIndex()
                            + ", actual " + current);
        }

        ObjectNode body = promotionBody(config.alias(), config.index(), current);
        if (!config.dryRun()) {
            http.requireJson(
                    "POST",
                    base + "/_aliases",
                    body,
                    "",
                    2,
                    Duration.ofSeconds(30));
        }
        ObjectNode row = MAPPER.createObjectNode();
        row.put("action", "promote");
        row.put("alias", config.alias());
        row.put("target_index", config.index());
        row.set("previous_indices", MAPPER.valueToTree(current));
        row.put("dry_run", config.dryRun());
        row.set("readiness", readiness);
        row.set("request", body);
        return row;
    }

    static ObjectNode targetReadiness(JsonHttpClient http, Config config) throws Exception {
        String base = stripTrailingSlash(config.esUrl());
        JsonNode health = http.requireJson(
                "GET",
                base + "/_cluster/health/" + config.index()
                        + "?wait_for_status=yellow&timeout=" + config.healthTimeoutSeconds() + "s",
                null,
                "",
                1,
                Duration.ofSeconds(config.healthTimeoutSeconds() + 5L));
        JsonNode count = http.requireJson(
                "GET",
                base + "/" + config.index() + "/_count",
                null,
                "",
                1,
                Duration.ofSeconds(30));
        return validateTargetReadiness(health, count, config.minimumDocuments());
    }

    static ObjectNode validateTargetReadiness(
            JsonNode health,
            JsonNode count,
            int minimumDocuments) {
        String status = health.path("status").asText("");
        boolean timedOut = health.path("timed_out").asBoolean(false);
        int activePrimaryShards = health.path("active_primary_shards").asInt(0);
        long documents = count.path("count").asLong(-1L);
        if (timedOut || "red".equals(status) || activePrimaryShards <= 0) {
            throw new IllegalStateException(
                    "target index is not ready: status=" + status
                            + ", timed_out=" + timedOut
                            + ", active_primary_shards=" + activePrimaryShards);
        }
        if (documents < minimumDocuments) {
            throw new IllegalStateException(
                    "target index document count " + documents
                            + " is below minimum " + minimumDocuments);
        }
        ObjectNode row = MAPPER.createObjectNode();
        row.put("status", status);
        row.put("timed_out", timedOut);
        row.put("active_primary_shards", activePrimaryShards);
        row.put("documents", documents);
        row.put("minimum_documents", minimumDocuments);
        return row;
    }

    static ObjectNode promotionBody(String alias, String targetIndex, List<String> currentIndices) {
        ObjectNode body = MAPPER.createObjectNode();
        ArrayNode actions = body.putArray("actions");
        for (String index : currentIndices) {
            if (!index.equals(targetIndex)) {
                actions.addObject().putObject("remove").put("index", index).put("alias", alias);
            }
        }
        ObjectNode add = actions.addObject().putObject("add");
        add.put("index", targetIndex);
        add.put("alias", alias);
        add.put("is_write_index", true);
        return body;
    }

    private static List<String> aliasIndices(JsonHttpClient http, String esUrl, String alias) throws Exception {
        String url = stripTrailingSlash(esUrl) + "/_alias/" + alias;
        JsonHttpClient.Response response = http.send(
                "GET", url, null, "application/json", "", 1, Duration.ofSeconds(15));
        if (response.status() == 404) {
            return List.of();
        }
        if (!response.successful()) {
            throw new IllegalStateException("cannot inspect alias, HTTP " + response.status() + ": " + response.body());
        }
        JsonNode payload = response.body().isBlank() ? MAPPER.createObjectNode() : MAPPER.readTree(response.body());
        List<String> indices = new ArrayList<>();
        payload.fieldNames().forEachRemaining(indices::add);
        return indices.stream().sorted().toList();
    }

    private static String stripTrailingSlash(String value) {
        return value.endsWith("/") ? value.substring(0, value.length() - 1) : value;
    }

    record Config(
            String action,
            String esUrl,
            String alias,
            String index,
            String expectedCurrentIndex,
            int minimumDocuments,
            int healthTimeoutSeconds,
            boolean dryRun) {

        static Config parse(String[] args) {
            Arguments values = Arguments.parse(args);
            String action = values.oneOf("action", "status", Set.of("status", "promote"));
            return new Config(
                    action,
                    values.string("es-url", "http://127.0.0.1:19200"),
                    values.required("alias"),
                    values.string("index", ""),
                    values.string("expected-current-index", ""),
                    values.nonNegativeInt("minimum-documents", 1),
                    values.positiveInt("health-timeout-seconds", 30),
                    values.bool("dry-run", false));
        }
    }
}
