package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

final class EmbeddingClient {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final JsonHttpClient http;
    private final Config config;

    EmbeddingClient(Config config) {
        this.http = new JsonHttpClient();
        this.config = config.validated();
    }

    List<List<Double>> embed(List<String> texts, InputType inputType)
            throws IOException, InterruptedException {
        if (texts.isEmpty()) {
            return List.of();
        }
        JsonNode response = http.requireJson(
                "POST",
                config.url(),
                requestBody(texts, inputType),
                config.apiKey(),
                config.maxRetries(),
                config.timeout());
        JsonNode rows = "dashscope".equals(config.apiFormat())
                ? response.path("output").path("embeddings")
                : response.path("data");
        if (!rows.isArray() || rows.size() != texts.size()) {
            throw new IllegalStateException(
                    "embedding response count mismatch: expected " + texts.size() + ", got " + rows.size());
        }

        List<VectorRow> ordered = new ArrayList<>(rows.size());
        for (int position = 0; position < rows.size(); position++) {
            JsonNode row = rows.get(position);
            int index = row.path("index").isNumber() ? row.path("index").asInt() : position;
            JsonNode vectorNode = row.path("embedding");
            if (!vectorNode.isArray() || vectorNode.size() != config.dimension()) {
                throw new IllegalStateException(
                        "embedding dimension mismatch at row " + position + ": expected "
                                + config.dimension() + ", got " + vectorNode.size());
            }
            List<Double> vector = MAPPER.convertValue(vectorNode, new TypeReference<>() { });
            ordered.add(new VectorRow(index, vector));
        }
        ordered.sort(java.util.Comparator.comparingInt(VectorRow::index));
        return ordered.stream().map(VectorRow::vector).toList();
    }

    ObjectNode requestBody(List<String> texts, InputType inputType) {
        ObjectNode body = MAPPER.createObjectNode();
        body.put("model", config.model());
        if ("dashscope".equals(config.apiFormat())) {
            ArrayNode input = body.putObject("input").putArray("texts");
            texts.forEach(input::add);
            ObjectNode parameters = body.putObject("parameters");
            parameters.put("text_type", inputType == InputType.QUERY ? "query" : "document");
            parameters.put("dimension", config.dimension());
            parameters.put("output_type", "dense");
            if (inputType == InputType.QUERY && !config.queryInstruction().isBlank()) {
                parameters.put("instruct", config.queryInstruction());
            }
            return body;
        }

        ArrayNode input = body.putArray("input");
        texts.stream()
                .map(text -> queryText(text, inputType))
                .forEach(input::add);
        body.put("encoding_format", "float");
        if ("openai".equals(config.apiFormat())) {
            body.put("dimensions", config.dimension());
        } else {
            body.put("input_type", inputType == InputType.QUERY ? "query" : "passage");
            body.put("dimension", config.dimension());
        }
        return body;
    }

    private String queryText(String text, InputType inputType) {
        if (inputType != InputType.QUERY || config.queryInstruction().isBlank()) {
            return text;
        }
        return "Instruct: " + config.queryInstruction() + "\nQuery:" + text;
    }

    enum InputType {
        DOCUMENT,
        QUERY
    }

    record Config(
            String url,
            String model,
            int dimension,
            String apiFormat,
            String apiKey,
            String queryInstruction,
            int maxRetries,
            Duration timeout) {

        Config validated() {
            if (url == null || url.isBlank() || model == null || model.isBlank()) {
                throw new IllegalArgumentException("embedding URL and model are required");
            }
            if (dimension <= 0 || maxRetries < 0 || timeout.isNegative() || timeout.isZero()) {
                throw new IllegalArgumentException("invalid embedding numeric configuration");
            }
            if (!Set.of("local", "openai", "dashscope").contains(apiFormat)) {
                throw new IllegalArgumentException("embedding API format must be local, openai, or dashscope");
            }
            if (Set.of("openai", "dashscope").contains(apiFormat) && (apiKey == null || apiKey.isBlank())) {
                throw new IllegalArgumentException("cloud embedding API key is required");
            }
            return this;
        }
    }

    private record VectorRow(int index, List<Double> vector) {
    }
}
