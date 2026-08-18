package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;

final class JsonHttpClient {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final HttpClient client;

    JsonHttpClient() {
        this.client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .version(HttpClient.Version.HTTP_1_1)
                .build();
    }

    Response send(
            String method,
            String url,
            byte[] body,
            String contentType,
            String bearerToken,
            int maxRetries,
            Duration timeout) throws IOException, InterruptedException {
        for (int attempt = 0; attempt <= maxRetries; attempt++) {
            HttpRequest.BodyPublisher publisher = body == null
                    ? HttpRequest.BodyPublishers.noBody()
                    : HttpRequest.BodyPublishers.ofByteArray(body);
            HttpRequest.Builder builder = HttpRequest.newBuilder(URI.create(url))
                    .timeout(timeout)
                    .version(HttpClient.Version.HTTP_1_1)
                    .method(method, publisher)
                    .header("Accept", "application/json");
            if (body != null) {
                builder.header("Content-Type", contentType);
            }
            if (bearerToken != null && !bearerToken.isBlank()) {
                builder.header("Authorization", "Bearer " + bearerToken);
            }

            try {
                HttpResponse<String> response = client.send(
                        builder.build(),
                        HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
                if (isRetryable(response.statusCode()) && attempt < maxRetries) {
                    sleep(attempt);
                    continue;
                }
                return new Response(response.statusCode(), response.body());
            } catch (IOException exception) {
                if (attempt >= maxRetries) {
                    throw exception;
                }
                sleep(attempt);
            }
        }
        throw new IllegalStateException("HTTP retry loop exhausted");
    }

    JsonNode requireJson(
            String method,
            String url,
            JsonNode body,
            String bearerToken,
            int maxRetries,
            Duration timeout) throws IOException, InterruptedException {
        byte[] bytes = body == null ? null : MAPPER.writeValueAsBytes(body);
        Response response = send(method, url, bytes, "application/json", bearerToken, maxRetries, timeout);
        if (!response.successful()) {
            throw new IllegalStateException("HTTP " + response.status() + " from " + url + ": " + response.body());
        }
        return response.body().isBlank() ? MAPPER.createObjectNode() : MAPPER.readTree(response.body());
    }

    JsonNode requireNdjson(
            String url,
            byte[] body,
            int maxRetries,
            Duration timeout) throws IOException, InterruptedException {
        Response response = send("POST", url, body, "application/x-ndjson", "", maxRetries, timeout);
        if (!response.successful()) {
            throw new IllegalStateException("HTTP " + response.status() + " from " + url + ": " + response.body());
        }
        return MAPPER.readTree(response.body());
    }

    private static boolean isRetryable(int status) {
        return status == 429 || status >= 500;
    }

    private static void sleep(int attempt) throws InterruptedException {
        Thread.sleep(1_000L << Math.min(attempt, 5));
    }

    record Response(int status, String body) {
        boolean successful() {
            return status >= 200 && status < 300;
        }
    }
}
