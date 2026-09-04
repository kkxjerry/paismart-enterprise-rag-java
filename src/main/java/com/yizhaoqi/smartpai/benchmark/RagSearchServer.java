package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.http.HttpClient;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Duration;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;

/** Authenticated online search API backed by the benchmark retrieval engine. */
public final class RagSearchServer {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Set<String> SERVER_ARGUMENTS = Set.of(
            "host", "port", "server-api-key-env", "allow-unauthenticated-loopback",
            "allow-body-principal", "max-request-bytes", "threads",
            "cache-ttl-seconds", "cache-max-entries");

    private RagSearchServer() {
    }

    public static void main(String[] args) throws Exception {
        ServerArguments serverArguments = ServerArguments.parse(args);
        EnterpriseRagJavaBenchmark.Config retrieval = EnterpriseRagJavaBenchmark.Config.parseOnline(
                stripServerArguments(args));
        if (!retrieval.evidenceEnabled()) {
            throw new IllegalArgumentException("online search requires evidence-enabled=true");
        }
        String apiKey = System.getenv().getOrDefault(serverArguments.apiKeyEnvironment(), "");
        validateAuthentication(serverArguments, apiKey);
        validateOnlineIndex(new JsonHttpClient(), retrieval);
        SearchApplication application = new SearchApplication(retrieval, serverArguments, apiKey);
        HttpServer server = HttpServer.create(
                new InetSocketAddress(serverArguments.host(), serverArguments.port()),
                128);
        server.createContext("/health", application::health);
        server.createContext("/metrics", application::metrics);
        server.createContext("/v1/search", application::search);
        server.setExecutor(Executors.newFixedThreadPool(serverArguments.threads()));
        server.start();
        System.out.println(MAPPER.writeValueAsString(Map.of(
                "event", "rag_search_server_started",
                "host", serverArguments.host(),
                "port", serverArguments.port(),
                "index", retrieval.index(),
                "auth_enabled", !apiKey.isBlank(),
                "cache_ttl_seconds", serverArguments.cacheTtlSeconds())));
    }

    static void validateAuthentication(ServerArguments arguments, String apiKey) {
        if (!apiKey.isBlank()) {
            return;
        }
        if (!arguments.allowUnauthenticatedLoopback()) {
            throw new IllegalArgumentException(
                    "online search requires a non-empty API key; set --allow-unauthenticated-loopback true only for local development");
        }
        if (!Set.of("127.0.0.1", "localhost", "::1").contains(arguments.host())) {
            throw new IllegalArgumentException(
                    "unauthenticated online search is allowed only on a loopback host");
        }
    }

    static void validateOnlineIndex(
            JsonHttpClient http,
            EnterpriseRagJavaBenchmark.Config config) throws Exception {
        String base = config.esUrl().endsWith("/")
                ? config.esUrl().substring(0, config.esUrl().length() - 1)
                : config.esUrl();
        JsonNode response = http.requireJson(
                "GET",
                base + "/" + config.index() + "/_mapping",
                null,
                "",
                1,
                Duration.ofSeconds(30));
        JsonNode index = response.path(config.index());
        if (index.isMissingNode() && response.fields().hasNext()) {
            index = response.fields().next().getValue();
        }
        validateOnlineMappings(index.path("mappings"));
    }

    static void validateOnlineMappings(JsonNode mappings) {
        JsonNode properties = mappings.path("properties");
        for (String field : List.of(
                "tenantId", "sourceType", "classification", "allowedGroupIds",
                "deniedGroupIds", "aclHash", "documentHash", "contentHash")) {
            if (!"keyword".equals(properties.path(field).path("type").asText())) {
                throw new IllegalStateException(
                        "online RAG index requires " + field + " mapped as keyword");
            }
        }
        if (!"date".equals(properties.path("deletedAt").path("type").asText())) {
            throw new IllegalStateException("online RAG index requires deletedAt mapped as date");
        }
    }

    static String[] stripServerArguments(String[] args) {
        List<String> output = new ArrayList<>();
        for (int index = 0; index < args.length; index++) {
            String argument = args[index];
            if (!argument.startsWith("--")) {
                output.add(argument);
                continue;
            }
            String name = argument.substring(2);
            if (SERVER_ARGUMENTS.contains(name)) {
                if (index + 1 >= args.length) {
                    throw new IllegalArgumentException("missing value for " + argument);
                }
                index++;
                continue;
            }
            output.add(argument);
            if (index + 1 < args.length && !args[index + 1].startsWith("--")) {
                output.add(args[++index]);
            }
        }
        return output.toArray(String[]::new);
    }

    static final class SearchApplication {
        private final EnterpriseRagJavaBenchmark.Config config;
        private final ServerArguments server;
        private final String apiKey;
        private final HttpClient client;
        private final ConcurrentHashMap<CacheKey, CacheEntry> cache = new ConcurrentHashMap<>();
        private final AtomicLong requests = new AtomicLong();
        private final AtomicLong errors = new AtomicLong();
        private final AtomicLong cacheHits = new AtomicLong();
        private final AtomicLong totalLatencyMicros = new AtomicLong();

        SearchApplication(
                EnterpriseRagJavaBenchmark.Config config,
                ServerArguments server,
                String apiKey) {
            this.config = config;
            this.server = server;
            this.apiKey = apiKey;
            this.client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
        }

        void health(HttpExchange exchange) throws IOException {
            if (!"GET".equals(exchange.getRequestMethod())) {
                json(exchange, 405, Map.of("error", "method_not_allowed"));
                return;
            }
            json(exchange, 200, Map.of(
                    "status", "ok",
                    "time", OffsetDateTime.now(ZoneOffset.UTC).toString()));
        }

        void metrics(HttpExchange exchange) throws IOException {
            if (!authorized(exchange)) {
                return;
            }
            long count = requests.get();
            json(exchange, 200, Map.of(
                    "requests", count,
                    "errors", errors.get(),
                    "cache_hits", cacheHits.get(),
                    "cache_entries", cache.size(),
                    "avg_latency_ms", count == 0 ? 0.0d : totalLatencyMicros.get() / 1000.0d / count));
        }

        void search(HttpExchange exchange) throws IOException {
            long started = System.nanoTime();
            requests.incrementAndGet();
            if (!authorized(exchange)) {
                return;
            }
            if (!"POST".equals(exchange.getRequestMethod())) {
                json(exchange, 405, Map.of("error", "method_not_allowed"));
                return;
            }
            try {
                byte[] body = exchange.getRequestBody().readNBytes(server.maxRequestBytes() + 1);
                if (body.length > server.maxRequestBytes()) {
                    json(exchange, 413, Map.of("error", "request_too_large"));
                    return;
                }
                JsonNode request = MAPPER.readTree(body);
                String query = request.path("query").asText("").trim();
                if (query.isBlank() || query.length() > 8_000) {
                    throw new IllegalArgumentException("query must contain 1..8000 characters");
                }
                PrincipalAssertion assertion = principalAssertion(
                        exchange.getRequestHeaders(),
                        request,
                        server.allowBodyPrincipal());
                SearchPrincipal principal = assertion.principal();
                List<String> sourceTypes = assertion.sourceTypes();
                int maxContexts = request.path("max_contexts").asInt(config.evidenceTopDocuments()
                        * config.evidenceChunksPerDocument());
                maxContexts = Math.max(1, Math.min(maxContexts, 100));
                CacheKey key = new CacheKey(
                        query,
                        principal.tenantId(),
                        principal.groupIds(),
                        principal.classifications(),
                        sourceTypes.stream().sorted().toList(),
                        config.index(),
                        maxContexts);
                Map<String, Object> cached = cached(key);
                String traceId = UUID.randomUUID().toString();
                if (cached != null) {
                    cacheHits.incrementAndGet();
                    Map<String, Object> response = new LinkedHashMap<>(cached);
                    response.put("trace_id", traceId);
                    response.put("cached", true);
                    json(exchange, 200, response);
                    return;
                }

                ObjectNode filter = EnterpriseRagJavaBenchmark.productionAclFilter(principal, sourceTypes);
                EnterpriseRagJavaBenchmark.Result result = EnterpriseRagJavaBenchmark.retrieveOnline(
                        client,
                        config,
                        query,
                        filter);
                Map<String, Object> response = new LinkedHashMap<>(
                        result.toOnlineResponse(traceId, query, principal, maxContexts));
                response.put("cached", false);
                response.put("served_at", OffsetDateTime.now(ZoneOffset.UTC).toString());
                putCache(key, response);
                json(exchange, 200, response);
            } catch (IllegalArgumentException exception) {
                errors.incrementAndGet();
                json(exchange, 400, Map.of("error", "invalid_request", "message", exception.getMessage()));
            } catch (Exception exception) {
                errors.incrementAndGet();
                json(exchange, 500, Map.of(
                        "error", "search_failed",
                        "type", exception.getClass().getName(),
                        "message", String.valueOf(exception.getMessage())));
            } finally {
                totalLatencyMicros.addAndGet((System.nanoTime() - started) / 1_000L);
            }
        }

        private boolean authorized(HttpExchange exchange) throws IOException {
            if (apiKey.isBlank()) {
                return true;
            }
            String authorization = exchange.getRequestHeaders().getFirst("Authorization");
            byte[] expected = ("Bearer " + apiKey).getBytes(StandardCharsets.UTF_8);
            byte[] actual = (authorization == null ? "" : authorization).getBytes(StandardCharsets.UTF_8);
            if (!MessageDigest.isEqual(expected, actual)) {
                json(exchange, 401, Map.of("error", "unauthorized"));
                return false;
            }
            return true;
        }

        private Map<String, Object> cached(CacheKey key) {
            CacheEntry entry = cache.get(key);
            if (entry == null) {
                return null;
            }
            if (entry.expiresAtEpochMs() < System.currentTimeMillis()) {
                cache.remove(key, entry);
                return null;
            }
            return entry.value();
        }

        private void putCache(CacheKey key, Map<String, Object> value) {
            if (server.cacheTtlSeconds() <= 0) {
                return;
            }
            if (cache.size() >= server.cacheMaxEntries()) {
                cache.entrySet().stream()
                        .min(Comparator.comparingLong(item -> item.getValue().expiresAtEpochMs()))
                        .ifPresent(item -> cache.remove(item.getKey(), item.getValue()));
            }
            Map<String, Object> stored = new LinkedHashMap<>(value);
            stored.remove("trace_id");
            stored.remove("cached");
            cache.put(key, new CacheEntry(
                    Map.copyOf(stored),
                    System.currentTimeMillis() + server.cacheTtlSeconds() * 1_000L));
        }
    }

    static PrincipalAssertion principalAssertion(
            Headers headers,
            JsonNode request,
            boolean allowBodyPrincipal) {
        if (allowBodyPrincipal) {
            JsonNode principalNode = request.path("principal");
            return new PrincipalAssertion(
                    new SearchPrincipal(
                            principalNode.path("tenant_id").asText(""),
                            strings(principalNode.path("group_ids")),
                            strings(principalNode.path("classifications"))),
                    strings(request.path("source_types")));
        }
        return new PrincipalAssertion(
                new SearchPrincipal(
                        header(headers, "X-RAG-Tenant-Id"),
                        headerValues(headers, "X-RAG-Group-Ids"),
                        headerValues(headers, "X-RAG-Classifications")),
                headerValues(headers, "X-RAG-Source-Types"));
    }

    private static String header(Headers headers, String name) {
        String value = headers == null ? null : headers.getFirst(name);
        return value == null ? "" : value.trim();
    }

    private static List<String> headerValues(Headers headers, String name) {
        if (headers == null) {
            return List.of();
        }
        List<String> values = new ArrayList<>();
        for (String line : headers.getOrDefault(name, List.of())) {
            for (String value : line.split(",")) {
                String normalized = value.trim();
                if (!normalized.isBlank()) {
                    values.add(normalized);
                }
            }
        }
        return values.stream().distinct().sorted().toList();
    }

    private static List<String> strings(JsonNode node) {
        List<String> values = new ArrayList<>();
        if (node.isTextual()) {
            values.add(node.asText());
        } else if (node.isArray()) {
            node.forEach(value -> {
                String text = value.asText("").trim();
                if (!text.isBlank()) {
                    values.add(text);
                }
            });
        }
        return values.stream().distinct().sorted().toList();
    }

    private static void json(HttpExchange exchange, int status, Object value) throws IOException {
        byte[] body = MAPPER.writeValueAsBytes(value);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.getResponseHeaders().set("Cache-Control", "no-store");
        exchange.sendResponseHeaders(status, body.length);
        exchange.getResponseBody().write(body);
        exchange.close();
    }

    record PrincipalAssertion(SearchPrincipal principal, List<String> sourceTypes) {

        PrincipalAssertion {
            sourceTypes = sourceTypes == null ? List.of() : sourceTypes.stream()
                    .map(String::trim)
                    .filter(value -> !value.isBlank())
                    .distinct()
                    .sorted()
                    .toList();
        }
    }

    record CacheKey(
            String query,
            String tenantId,
            List<String> groupIds,
            List<String> classifications,
            List<String> sourceTypes,
            String index,
            int maxContexts) {
    }

    record CacheEntry(Map<String, Object> value, long expiresAtEpochMs) {
    }

    record ServerArguments(
            String host,
            int port,
            String apiKeyEnvironment,
            boolean allowUnauthenticatedLoopback,
            boolean allowBodyPrincipal,
            int maxRequestBytes,
            int threads,
            long cacheTtlSeconds,
            int cacheMaxEntries) {

        static ServerArguments parse(String[] args) {
            Arguments values = Arguments.parse(args);
            long cacheTtlSeconds = Long.parseLong(values.string("cache-ttl-seconds", "0"));
            if (cacheTtlSeconds < 0) {
                throw new IllegalArgumentException("--cache-ttl-seconds must not be negative");
            }
            return new ServerArguments(
                    values.string("host", "127.0.0.1"),
                    values.positiveInt("port", 18090),
                    values.string("server-api-key-env", "RAG_SEARCH_API_KEY"),
                    values.bool("allow-unauthenticated-loopback", false),
                    values.bool("allow-body-principal", false),
                    values.positiveInt("max-request-bytes", 1_048_576),
                    values.positiveInt("threads", Math.max(4, Runtime.getRuntime().availableProcessors())),
                    cacheTtlSeconds,
                    values.positiveInt("cache-max-entries", 2_000));
        }
    }
}
