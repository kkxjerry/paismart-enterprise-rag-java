package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.nio.charset.StandardCharsets;
import java.time.Duration;

public final class ElasticsearchIndexCommand {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private ElasticsearchIndexCommand() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        JsonHttpClient http = new JsonHttpClient();
        String indexUrl = stripTrailingSlash(config.esUrl()) + "/" + config.index();
        JsonHttpClient.Response existing = http.send(
                "HEAD", indexUrl, null, "application/json", "", 1, Duration.ofSeconds(15));
        if (existing.status() == 200) {
            if (config.skipExisting()) {
                System.out.println("{\"event\":\"index_exists\",\"index\":\"" + config.index() + "\"}");
                return;
            }
            throw new IllegalStateException(
                    "index already exists: " + config.index() + "; use a new name or --skip-existing true");
        }
        if (existing.status() != 404) {
            throw new IllegalStateException(
                    "cannot inspect index, HTTP " + existing.status() + ": " + existing.body());
        }

        ObjectNode mapping = buildMapping(config);
        http.requireJson("PUT", indexUrl, mapping, "", 1, Duration.ofSeconds(30));
        ObjectNode summary = MAPPER.createObjectNode();
        summary.put("event", "index_created");
        summary.put("index", config.index());
        summary.put("embedding_model", config.embeddingModel());
        summary.put("embedding_dimension", config.embeddingDimension());
        summary.put("bm25_k1", config.bm25K1());
        summary.put("bm25_b", config.bm25B());
        System.out.println(MAPPER.writeValueAsString(summary));
    }

    static ObjectNode buildMapping(Config config) {
        ObjectNode root = MAPPER.createObjectNode();
        ObjectNode settings = root.putObject("settings");
        settings.put("number_of_shards", config.shards());
        settings.put("number_of_replicas", config.replicas());
        settings.put("refresh_interval", config.refreshInterval());
        ObjectNode similarity = settings.putObject("similarity").putObject("enterprise_bm25");
        similarity.put("type", "BM25");
        similarity.put("k1", config.bm25K1());
        similarity.put("b", config.bm25B());

        ObjectNode mappings = root.putObject("mappings");
        mappings.put("dynamic", "strict");
        ObjectNode meta = mappings.putObject("_meta");
        meta.put("schema_version", "1-enterpriserag-java");
        meta.put("dataset", "onyx-dot-app/EnterpriseRAG-Bench");
        meta.put("embedding_model", config.embeddingModel());
        meta.put("embedding_dimension", config.embeddingDimension());
        meta.put("bm25_k1", config.bm25K1());
        meta.put("bm25_b", config.bm25B());

        ObjectNode properties = mappings.putObject("properties");
        keyword(properties, "benchmarkDocId");
        properties.putObject("chunkId").put("type", "integer");
        text(properties, "title", true);
        text(properties, "textContent", false);
        keyword(properties, "sourceType");
        properties.putObject("sourcePath").put("type", "keyword").put("ignore_above", 1024);
        keyword(properties, "sourceDataset");
        keyword(properties, "tenantId");
        keyword(properties, "classification");
        keyword(properties, "allowedGroupIds");
        keyword(properties, "deniedGroupIds");
        ObjectNode vector = properties.putObject("vector");
        vector.put("type", "dense_vector");
        vector.put("dims", config.embeddingDimension());
        vector.put("index", true);
        vector.put("similarity", "cosine");
        keyword(properties, "modelVersion");
        keyword(properties, "contentHash");
        properties.putObject("indexedAt").put("type", "date");
        return root;
    }

    private static void keyword(ObjectNode properties, String name) {
        properties.putObject(name).put("type", "keyword");
    }

    private static void text(ObjectNode properties, String name, boolean keywordSubfield) {
        ObjectNode field = properties.putObject(name);
        field.put("type", "text");
        field.put("analyzer", "standard");
        field.put("search_analyzer", "standard");
        field.put("similarity", "enterprise_bm25");
        ObjectNode subfields = field.putObject("fields");
        if (keywordSubfield) {
            subfields.putObject("keyword").put("type", "keyword").put("ignore_above", 512);
        }
        ObjectNode english = subfields.putObject("english");
        english.put("type", "text");
        english.put("analyzer", "english");
        english.put("search_analyzer", "english");
        english.put("similarity", "enterprise_bm25");
    }

    private static String stripTrailingSlash(String value) {
        return value.endsWith("/") ? value.substring(0, value.length() - 1) : value;
    }

    record Config(
            String esUrl,
            String index,
            String embeddingModel,
            int embeddingDimension,
            int shards,
            int replicas,
            String refreshInterval,
            double bm25K1,
            double bm25B,
            boolean skipExisting) {

        static Config parse(String[] args) {
            Arguments values = Arguments.parse(args);
            return new Config(
                    values.string("es-url", "http://127.0.0.1:19200"),
                    values.required("index"),
                    values.required("embedding-model"),
                    values.positiveInt("embedding-dimension", 2048),
                    values.positiveInt("shards", 1),
                    values.nonNegativeInt("replicas", 0),
                    values.string("refresh-interval", "30s"),
                    Double.parseDouble(values.string("bm25-k1", "2.2")),
                    Double.parseDouble(values.string("bm25-b", "1.0")),
                    values.bool("skip-existing", false));
        }
    }
}
