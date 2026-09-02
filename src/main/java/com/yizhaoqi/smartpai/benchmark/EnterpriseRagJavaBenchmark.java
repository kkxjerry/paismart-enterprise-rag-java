package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.yizhaoqi.smartpai.service.Bm25QueryRewriter;
import com.yizhaoqi.smartpai.service.EvidenceBuilder;
import com.yizhaoqi.smartpai.service.ReciprocalRankFusion;

import java.io.BufferedWriter;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.http.HttpClient;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Runs the production-shaped Java Dense + BM25 + RRF retrieval path against
 * EnterpriseRAG without starting Spring or connecting to application databases.
 */
public final class EnterpriseRagJavaBenchmark {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Pattern TOKEN_PATTERN = Pattern.compile(
            "[a-z0-9_][a-z0-9_./:+#-]*|[\\u4e00-\\u9fff]",
            Pattern.CASE_INSENSITIVE);
    private static final Pattern KEY_TOKEN_PATTERN = Pattern.compile(
            "[a-z0-9]+(?:[._:/-][a-z0-9]+)*",
            Pattern.CASE_INSENSITIVE);
    private static final Pattern NUMBER_UNIT_PATTERN = Pattern.compile("^(\\d+)([a-z]+)$");
    private static final Set<String> FACT_STOPWORDS = Set.of(
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "in", "includes", "including", "is", "it", "of", "on", "or", "that",
            "the", "to", "with", "default", "defaults", "file", "files", "limit",
            "limits", "per", "request", "requests", "size", "total", "upload", "uploads");
    private static final EvidenceBuilder EVIDENCE_BUILDER = new EvidenceBuilder();
    private static final double FACT_COVERAGE_THRESHOLD = 0.60d;

    private EnterpriseRagJavaBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        if (Set.of("openai", "dashscope").contains(config.embeddingApiFormat())
                && config.embeddingApiKey().isBlank()) {
            throw new IllegalArgumentException(
                    "a cloud embedding API key is required via the configured environment variable "
                            + "or --embedding-api-key");
        }

        Map<String, Object> execution = new LinkedHashMap<>();
        execution.put("questions", config.questions().toString());
        execution.put("engine", config.engine());
        execution.put("es_url", config.esUrl());
        execution.put("index", config.index());
        execution.put("embedding_url", config.embeddingUrl());
        execution.put("embedding_api_format", config.embeddingApiFormat());
        execution.put("embedding_model", config.embeddingModel());
        execution.put("embedding_dimension", config.embeddingDimension());
        execution.put("embedding_query_instruction", config.embeddingQueryInstruction());
        execution.put("retrieval_mode", config.retrievalMode());
        execution.put("retriever_k", config.retrieverK());
        execution.put("dense_chunk_candidates", config.denseChunkCandidates());
        execution.put("dense_num_candidates", config.denseNumCandidates());
        execution.put("rrf_k", config.rrfK());
        execution.put("dense_weight", config.denseWeight());
        execution.put("bm25_weight", config.bm25Weight());
        execution.put("keyword_bm25_enabled", config.keywordBm25Enabled());
        execution.put("keyword_bm25_weight", config.keywordBm25Weight());
        execution.put("english_bm25_enabled", config.englishBm25Enabled());
        execution.put("english_bm25_weight", config.englishBm25Weight());
        execution.put("top_k", config.topK());
        execution.put("evidence_enabled", config.evidenceEnabled());
        execution.put("evidence_top_documents", config.evidenceTopDocuments());
        execution.put("evidence_candidate_chunks_per_document", config.evidenceCandidateChunksPerDocument());
        execution.put("evidence_chunks_per_document", config.evidenceChunksPerDocument());
        execution.put("evidence_token_budget", config.evidenceTokenBudget());
        execution.put("evidence_per_document_token_budget", config.evidencePerDocumentTokenBudget());
        execution.put("evidence_max_chunk_tokens", config.evidenceMaxChunkTokens());
        execution.put("evidence_redundancy_penalty", config.evidenceRedundancyPenalty());
        execution.put("progress_every", config.progressEvery());
        execution.put("summary_output", config.output().toString());
        execution.put("details_output", config.detailsOutput().toString());
        execution.put("manifest_output", config.manifestOutput().toString());
        execution.put("evidence_output", config.evidenceEnabled() ? config.evidenceOutput().toString() : null);

        RunManifest.Session manifest = RunManifest.start(
                config.manifestOutput(),
                config.runId(),
                config.experiment(),
                config.questions(),
                execution);
        try {
            manifest.setIndexMetadata(loadIndexMetadata(config));
            List<Question> questions = loadQuestions(config.questions());
            HttpClient client = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(10))
                    .build();

            createParent(config.output());
            createParent(config.detailsOutput());
            if (config.evidenceEnabled()) {
                createParent(config.evidenceOutput());
            }

            List<Result> results = new ArrayList<>(questions.size());
            long runStarted = System.nanoTime();
            BufferedWriter evidence = config.evidenceEnabled()
                    ? Files.newBufferedWriter(config.evidenceOutput(), StandardCharsets.UTF_8)
                    : null;
            try (BufferedWriter details = Files.newBufferedWriter(config.detailsOutput(), StandardCharsets.UTF_8);
                 BufferedWriter evidenceWriter = evidence) {
                for (int index = 0; index < questions.size(); index++) {
                    Question question = questions.get(index);
                    Result result = retrieve(client, config, question);
                    results.add(result);
                    details.write(MAPPER.writeValueAsString(result.toDetails(question)));
                    details.newLine();
                    if (evidenceWriter != null) {
                        evidenceWriter.write(MAPPER.writeValueAsString(result.toEvidenceRow(config, question)));
                        evidenceWriter.newLine();
                    }

                    int completed = index + 1;
                    if (config.progressEvery() > 0
                            && (completed % config.progressEvery() == 0 || completed == questions.size())) {
                        double seconds = (System.nanoTime() - runStarted) / 1_000_000_000.0d;
                        System.out.printf(
                                "{\"event\":\"java_retrieval_progress\",\"completed\":%d,\"total\":%d,\"qps\":%.2f}%n",
                                completed,
                                questions.size(),
                                completed / Math.max(seconds, 0.001d));
                    }
                }
            }

            Map<String, Object> summary = summarize(config, questions, results);
            summary.put("manifest_output", config.manifestOutput().toString());
            summary.put("details_output", config.detailsOutput().toString());
            summary.put("evidence_output", config.evidenceEnabled() ? config.evidenceOutput().toString() : null);
            MAPPER.writerWithDefaultPrettyPrinter().writeValue(config.output().toFile(), summary);
            manifest.complete(summary);
            System.out.println(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(summary));
        } catch (Exception exception) {
            manifest.fail(exception);
            throw exception;
        }
    }

    static Result retrieve(HttpClient client, Config config, Question question) throws Exception {
        long started = System.nanoTime();
        ObjectNode filter = sourceAclFilter(question.sourceTypes());
        List<RouteRanking> routes = new ArrayList<>();

        double embeddingMs = 0.0d;
        double denseMs = 0.0d;
        if (!"bm25".equals(config.retrievalMode())) {
            long embeddingStarted = System.nanoTime();
            List<Double> vector = embed(client, config, question.text());
            embeddingMs = elapsedMs(embeddingStarted);
            long denseStarted = System.nanoTime();
            List<RankedDocument> dense = denseSearch(client, config, vector, filter);
            denseMs = elapsedMs(denseStarted);
            routes.add(new RouteRanking("dense", config.denseWeight(), dense));
        }

        double bm25Ms = 0.0d;
        double keywordBm25Ms = 0.0d;
        double englishBm25Ms = 0.0d;
        if (!"dense".equals(config.retrievalMode())) {
            long bm25Started = System.nanoTime();
            List<RankedDocument> bm25 = bm25Search(
                    client,
                    config,
                    question.text(),
                    filter,
                    standardBm25Fields());
            bm25Ms = elapsedMs(bm25Started);
            routes.add(new RouteRanking("bm25_original", config.bm25Weight(), bm25));

            String keywordQuery = Bm25QueryRewriter.keywordQuery(question.text());
            if (config.keywordBm25Enabled()
                    && Bm25QueryRewriter.isDistinctVariant(question.text(), keywordQuery)) {
                long keywordStarted = System.nanoTime();
                List<RankedDocument> keywordBm25 = bm25Search(
                        client,
                        config,
                        keywordQuery,
                        filter,
                        standardBm25Fields());
                keywordBm25Ms = elapsedMs(keywordStarted);
                if (!keywordBm25.isEmpty()) {
                    routes.add(new RouteRanking("bm25_keyword", config.keywordBm25Weight(), keywordBm25));
                }
            }
            if (config.englishBm25Enabled()) {
                long englishStarted = System.nanoTime();
                List<RankedDocument> englishBm25 = bm25Search(
                        client,
                        config,
                        question.text(),
                        filter,
                        englishBm25Fields());
                englishBm25Ms = elapsedMs(englishStarted);
                if (!englishBm25.isEmpty()) {
                    routes.add(new RouteRanking("bm25_english", config.englishBm25Weight(), englishBm25));
                }
            }
        }

        long fusionStarted = System.nanoTime();
        FusionResult fusion = fuseRoutes(routes, config);
        double fusionMs = elapsedMs(fusionStarted);
        double retrievalLatencyMs = elapsedMs(started);

        double evidenceMs = 0.0d;
        EvidenceBuilder.EvidenceBundle evidence = new EvidenceBuilder.EvidenceBundle(
                List.of(), 0, List.of(), 0, 0);
        if (config.evidenceEnabled() && !fusion.documents().isEmpty()) {
            long evidenceStarted = System.nanoTime();
            Map<String, List<RankedDocument>> lexicalChunks = evidenceChunksSearch(
                    client,
                    config,
                    question.text(),
                    filter,
                    fusion.documents());
            List<EvidenceBuilder.DocumentCandidate> candidates = buildEvidenceCandidates(
                    config,
                    fusion,
                    lexicalChunks);
            evidence = EVIDENCE_BUILDER.build(question.text(), candidates, config.evidenceConfig());
            evidenceMs = elapsedMs(evidenceStarted);
        }
        EvidenceScores evidenceScores = config.evidenceEnabled()
                ? scoreEvidence(question, evidence)
                : new EvidenceScores(null, null, null);
        return new Result(
                fusion.documents(),
                fusion.routeEvidenceByDoc(),
                evidence,
                evidenceScores,
                embeddingMs,
                denseMs,
                bm25Ms,
                keywordBm25Ms,
                englishBm25Ms,
                fusionMs,
                evidenceMs,
                retrievalLatencyMs,
                elapsedMs(started));
    }

    private static FusionResult fuseRoutes(List<RouteRanking> routes, Config config) {
        if (routes.isEmpty()) {
            throw new IllegalStateException("at least one retrieval route is required");
        }
        Map<String, List<RouteEvidence>> routeEvidenceByDoc = new LinkedHashMap<>();
        for (RouteRanking route : routes) {
            for (int index = 0; index < route.documents().size(); index++) {
                RankedDocument document = route.documents().get(index);
                int rank = index + 1;
                EvidenceBuilder.RouteSignal signal = new EvidenceBuilder.RouteSignal(
                        route.name(),
                        rank,
                        route.weight(),
                        document.score(),
                        route.weight() / (config.rrfK() + rank),
                        document.chunkEsId());
                routeEvidenceByDoc.computeIfAbsent(document.docId(), ignored -> new ArrayList<>())
                        .add(new RouteEvidence(signal, document));
            }
        }

        List<RankedDocument> fused;
        if (routes.size() == 1) {
            fused = routes.get(0).documents().stream().limit(config.topK()).toList();
        } else {
            fused = ReciprocalRankFusion.fuse(
                            routes.stream().map(RouteRanking::documents).toList(),
                            routes.stream().map(RouteRanking::weight).toList(),
                            RankedDocument::docId,
                            config.rrfK(),
                            config.topK())
                    .stream()
                    .map(item -> item.value().withScore(item.score()))
                    .toList();
        }

        Map<String, List<RouteEvidence>> immutableEvidence = new LinkedHashMap<>();
        routeEvidenceByDoc.forEach((docId, evidence) ->
                immutableEvidence.put(docId, List.copyOf(evidence)));
        return new FusionResult(fused, Map.copyOf(immutableEvidence));
    }

    private static Map<String, List<RankedDocument>> evidenceChunksSearch(
            HttpClient client,
            Config config,
            String query,
            ObjectNode filter,
            List<RankedDocument> documents) throws Exception {
        List<String> docIds = documents.stream()
                .limit(config.evidenceTopDocuments())
                .map(RankedDocument::docId)
                .toList();
        if (docIds.isEmpty()) {
            return Map.of();
        }
        JsonNode response = post(
                client,
                searchUrl(config),
                evidenceSearchBody(config, query, filter, docIds));
        return parseEvidenceHits(response);
    }

    static ObjectNode evidenceSearchBody(
            Config config,
            String query,
            ObjectNode filter,
            List<String> docIds) {
        ObjectNode body = MAPPER.createObjectNode();
        body.put("size", Math.max(1, docIds.size()));
        body.put("track_total_hits", false);
        addSourceFields(body.putArray("_source"));

        ObjectNode bool = MAPPER.createObjectNode();
        ArrayNode filters = bool.putArray("filter");
        filters.add(termsQuery("benchmarkDocId", docIds));
        if (filter != null) {
            filters.add(filter);
        }
        ArrayNode should = bool.putArray("should");
        should.add(multiMatchQuery(query, standardBm25Fields(), 1.0d));
        String keywordQuery = Bm25QueryRewriter.keywordQuery(query);
        if (Bm25QueryRewriter.isDistinctVariant(query, keywordQuery)) {
            should.add(multiMatchQuery(keywordQuery, standardBm25Fields(), 1.20d));
        }
        if (config.englishBm25Enabled()) {
            should.add(multiMatchQuery(query, englishBm25Fields(), 1.0d));
        }
        bool.put("minimum_should_match", 0);
        body.set("query", MAPPER.createObjectNode().set("bool", bool));

        ObjectNode collapse = body.putObject("collapse");
        collapse.put("field", "benchmarkDocId");
        ObjectNode innerHits = collapse.putObject("inner_hits");
        innerHits.put("name", "evidence_chunks");
        innerHits.put("size", config.evidenceCandidateChunksPerDocument());
        innerHits.put("track_scores", true);
        ArrayNode sort = innerHits.putArray("sort");
        sort.add(MAPPER.createObjectNode().put("_score", "desc"));
        sort.add(MAPPER.createObjectNode().put("chunkId", "asc"));
        return body;
    }

    private static ObjectNode multiMatchQuery(String query, List<String> fields, double boost) {
        ObjectNode multiMatch = MAPPER.createObjectNode();
        multiMatch.put("query", query);
        fields.forEach(multiMatch.putArray("fields")::add);
        multiMatch.put("type", "best_fields");
        multiMatch.put("operator", "or");
        multiMatch.put("boost", boost);
        return MAPPER.createObjectNode().set("multi_match", multiMatch);
    }

    private static Map<String, List<RankedDocument>> parseEvidenceHits(JsonNode response) {
        Map<String, List<RankedDocument>> chunksByDocument = new LinkedHashMap<>();
        for (JsonNode hit : response.path("hits").path("hits")) {
            RankedDocument parentHit = parseHit(hit);
            if (parentHit == null) {
                continue;
            }
            JsonNode innerHits = hit.path("inner_hits").path("evidence_chunks").path("hits").path("hits");
            List<RankedDocument> chunks = new ArrayList<>();
            if (innerHits.isArray()) {
                for (JsonNode innerHit : innerHits) {
                    RankedDocument chunk = parseHit(innerHit);
                    if (chunk != null) {
                        chunks.add(chunk);
                    }
                }
            }
            if (chunks.isEmpty()) {
                chunks.add(parentHit);
            }
            chunksByDocument.put(parentHit.docId(), List.copyOf(chunks));
        }
        return Map.copyOf(chunksByDocument);
    }

    private static List<EvidenceBuilder.DocumentCandidate> buildEvidenceCandidates(
            Config config,
            FusionResult fusion,
            Map<String, List<RankedDocument>> lexicalChunks) {
        List<EvidenceBuilder.DocumentCandidate> candidates = new ArrayList<>();
        List<RankedDocument> topDocuments = fusion.documents().stream()
                .limit(config.evidenceTopDocuments())
                .toList();
        for (int index = 0; index < topDocuments.size(); index++) {
            RankedDocument document = topDocuments.get(index);
            List<RouteEvidence> routeEvidence = fusion.routeEvidenceByDoc()
                    .getOrDefault(document.docId(), List.of());
            List<EvidenceBuilder.RouteSignal> routeSignals = routeEvidence.stream()
                    .map(RouteEvidence::signal)
                    .toList();
            double evidenceFusionScore = routeSignals.stream()
                    .mapToDouble(EvidenceBuilder.RouteSignal::rrfContribution)
                    .sum();
            List<EvidenceBuilder.ChunkCandidate> chunks = new ArrayList<>();
            for (RouteEvidence evidence : routeEvidence) {
                chunks.add(toChunkCandidate(evidence.chunk(), 0.0d, List.of(evidence.signal())));
            }
            for (RankedDocument lexicalChunk : lexicalChunks.getOrDefault(document.docId(), List.of())) {
                chunks.add(toChunkCandidate(lexicalChunk, lexicalChunk.score(), List.of()));
            }
            candidates.add(new EvidenceBuilder.DocumentCandidate(
                    document.docId(),
                    index + 1,
                    evidenceFusionScore,
                    routeSignals,
                    chunks));
        }
        return List.copyOf(candidates);
    }

    private static EvidenceBuilder.ChunkCandidate toChunkCandidate(
            RankedDocument document,
            double lexicalScore,
            List<EvidenceBuilder.RouteSignal> routeSignals) {
        return new EvidenceBuilder.ChunkCandidate(
                document.docId(),
                document.chunkEsId(),
                document.chunkId(),
                document.sourceType(),
                document.sourcePath(),
                document.title(),
                document.text(),
                document.classification(),
                document.documentVersion(),
                document.documentHash(),
                document.sourceUpdatedAt(),
                document.contentHash(),
                lexicalScore,
                routeSignals);
    }

    private static EvidenceScores scoreEvidence(
            Question question,
            EvidenceBuilder.EvidenceBundle evidence) {
        String context = evidence.spans().stream()
                .map(EvidenceBuilder.EvidenceSpan::text)
                .reduce("", (left, right) -> left + "\n" + right);
        List<Double> factRecalls = question.answerFacts().stream()
                .map(fact -> tokenRecall(fact, context))
                .filter(value -> value != null)
                .toList();
        Double averageFactRecall = factRecalls.isEmpty()
                ? null
                : factRecalls.stream().mapToDouble(Double::doubleValue).average().orElse(0.0d);
        Double factCoverage = factRecalls.isEmpty()
                ? null
                : factRecalls.stream().filter(value -> value >= FACT_COVERAGE_THRESHOLD).count()
                        / (double) factRecalls.size();
        Double goldAnswerRecall = question.goldAnswer().isBlank()
                ? null
                : tokenRecall(question.goldAnswer(), context);
        return new EvidenceScores(averageFactRecall, factCoverage, goldAnswerRecall);
    }

    private static Double tokenRecall(String expected, String actual) {
        Set<String> expectedTokens = keyTokenSet(expected);
        if (expectedTokens.isEmpty()) {
            return null;
        }
        int expectedCount = expectedTokens.size();
        expectedTokens.retainAll(keyTokenSet(actual));
        return expectedTokens.size() / (double) expectedCount;
    }

    private static Set<String> keyTokenSet(String text) {
        Set<String> tokens = new HashSet<>();
        var matcher = KEY_TOKEN_PATTERN.matcher(
                text == null ? "" : text.toLowerCase(Locale.ROOT));
        while (matcher.find()) {
            String token = matcher.group();
            if (token.isBlank() || FACT_STOPWORDS.contains(token)) {
                continue;
            }
            if (token.length() <= 1 && !token.chars().allMatch(Character::isDigit)) {
                continue;
            }
            tokens.add(token);
            var numberUnit = NUMBER_UNIT_PATTERN.matcher(token);
            if (numberUnit.matches()) {
                tokens.add(numberUnit.group(1));
                tokens.add(numberUnit.group(2));
            }
        }
        return tokens;
    }

    private static List<Double> embed(HttpClient client, Config config, String query) throws Exception {
        JsonNode response = post(
                client,
                config.embeddingUrl(),
                embeddingRequestBody(config, query),
                config.embeddingApiKey());
        JsonNode vector = "dashscope".equals(config.embeddingApiFormat())
                ? response.path("output").path("embeddings").path(0).path("embedding")
                : response.path("data").path(0).path("embedding");
        if (!vector.isArray() || vector.size() != config.embeddingDimension()) {
            throw new IllegalStateException(
                    "embedding response dimension mismatch: expected "
                            + config.embeddingDimension() + ", got " + vector.size());
        }
        return MAPPER.convertValue(vector, new TypeReference<>() { });
    }

    static ObjectNode embeddingRequestBody(Config config, String query) {
        ObjectNode body = MAPPER.createObjectNode();
        body.put("model", config.embeddingModel());
        if ("dashscope".equals(config.embeddingApiFormat())) {
            body.putObject("input").putArray("texts").add(query);
            ObjectNode parameters = body.putObject("parameters");
            parameters.put("text_type", "query");
            parameters.put("dimension", config.embeddingDimension());
            parameters.put("output_type", "dense");
            if (!config.embeddingQueryInstruction().isBlank()) {
                parameters.put("instruct", config.embeddingQueryInstruction());
            }
        } else {
            String embeddingQuery = config.embeddingQueryInstruction().isBlank()
                    ? query
                    : "Instruct: " + config.embeddingQueryInstruction() + "\nQuery:" + query;
            body.putArray("input").add(embeddingQuery);
        }
        if ("openai".equals(config.embeddingApiFormat())) {
            body.put("dimensions", config.embeddingDimension());
        } else if ("local".equals(config.embeddingApiFormat())) {
            body.put("input_type", "query");
            body.put("dimension", config.embeddingDimension());
        }
        if (!"dashscope".equals(config.embeddingApiFormat())) {
            body.put("encoding_format", "float");
        }
        return body;
    }

    private static List<RankedDocument> denseSearch(
            HttpClient client,
            Config config,
            List<Double> vector,
            ObjectNode filter) throws Exception {
        return parseHits(post(client, searchUrl(config), denseSearchBody(config, vector, filter)));
    }

    static ObjectNode denseSearchBody(Config config, List<Double> vector, ObjectNode filter) {
        ObjectNode body = commonSearchBody(config);
        if ("opensearch".equals(config.engine())) {
            ObjectNode vectorQuery = MAPPER.createObjectNode();
            vectorQuery.set("vector", MAPPER.valueToTree(vector));
            vectorQuery.put("k", config.denseChunkCandidates());
            vectorQuery.putObject("method_parameters").put("ef_search", config.denseNumCandidates());
            if (filter != null) {
                vectorQuery.set("filter", filter);
            }
            ObjectNode knn = MAPPER.createObjectNode();
            knn.set("vector", vectorQuery);
            body.set("query", MAPPER.createObjectNode().set("knn", knn));
        } else {
            ObjectNode knn = body.putObject("knn");
            knn.put("field", "vector");
            knn.set("query_vector", MAPPER.valueToTree(vector));
            knn.put("k", config.denseChunkCandidates());
            knn.put("num_candidates", config.denseNumCandidates());
            if (filter != null) {
                knn.set("filter", filter);
            }
        }
        return body;
    }

    private static List<RankedDocument> bm25Search(
            HttpClient client,
            Config config,
            String query,
            ObjectNode filter,
            List<String> fields) throws Exception {
        return parseHits(post(client, searchUrl(config), bm25SearchBody(config, query, filter, fields)));
    }

    static ObjectNode bm25SearchBody(
            Config config,
            String query,
            ObjectNode filter,
            List<String> fields) {
        ObjectNode body = commonSearchBody(config);
        ObjectNode multiMatch = MAPPER.createObjectNode();
        multiMatch.put("query", query);
        ArrayNode fieldArray = multiMatch.putArray("fields");
        fields.forEach(fieldArray::add);
        multiMatch.put("type", "best_fields");
        multiMatch.put("operator", "or");

        if (filter == null) {
            body.set("query", MAPPER.createObjectNode().set("multi_match", multiMatch));
        } else {
            ObjectNode bool = MAPPER.createObjectNode();
            bool.putArray("must").add(MAPPER.createObjectNode().set("multi_match", multiMatch));
            bool.putArray("filter").add(filter);
            body.set("query", MAPPER.createObjectNode().set("bool", bool));
        }
        return body;
    }

    private static List<String> standardBm25Fields() {
        return List.of("title^2.0", "textContent^1.0");
    }

    private static List<String> englishBm25Fields() {
        return List.of("title.english^2.0", "textContent.english^1.0");
    }

    private static ObjectNode commonSearchBody(Config config) {
        ObjectNode body = MAPPER.createObjectNode();
        body.put("size", config.retrieverK());
        body.put("track_total_hits", false);
        body.putObject("collapse").put("field", "benchmarkDocId");
        addSourceFields(body.putArray("_source"));
        return body;
    }

    private static void addSourceFields(ArrayNode fields) {
        fields.add("benchmarkDocId")
                .add("chunkId")
                .add("title")
                .add("textContent")
                .add("sourceType")
                .add("sourcePath")
                .add("classification")
                .add("documentVersion")
                .add("documentHash")
                .add("sourceUpdatedAt")
                .add("contentHash")
                .add("indexedAt");
    }

    static ObjectNode sourceAclFilter(List<String> sourceTypes) {
        if (sourceTypes.isEmpty()) {
            return null;
        }
        ArrayNode filters = MAPPER.createArrayNode();
        filters.add(termQuery("tenantId", "tenant_redwood"));
        filters.add(termsQuery("sourceType", sourceTypes));
        filters.add(termsQuery(
                "allowedGroupIds",
                sourceTypes.stream().map(value -> "source:" + value).toList()));

        ObjectNode bool = MAPPER.createObjectNode();
        bool.set("filter", filters);
        bool.putArray("must_not").add(termsQuery(
                "deniedGroupIds",
                sourceTypes.stream().map(value -> "source:" + value).toList()));
        return MAPPER.createObjectNode().set("bool", bool);
    }

    private static ObjectNode termQuery(String field, String value) {
        return MAPPER.createObjectNode().set(
                "term",
                MAPPER.createObjectNode().put(field, value));
    }

    private static ObjectNode termsQuery(String field, List<String> values) {
        return MAPPER.createObjectNode().set(
                "terms",
                MAPPER.createObjectNode().set(field, MAPPER.valueToTree(values)));
    }

    private static String searchUrl(Config config) {
        return stripTrailingSlash(config.esUrl()) + "/" + config.index() + "/_search";
    }

    private static JsonNode post(HttpClient client, String url, JsonNode body) throws Exception {
        return post(client, url, body, "");
    }

    private static JsonNode post(
            HttpClient client,
            String url,
            JsonNode body,
            String bearerToken) throws Exception {
        byte[] requestBody = MAPPER.writeValueAsBytes(body);
        int maxAttempts = bearerToken.isBlank() ? 1 : 4;
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            HttpURLConnection connection = (HttpURLConnection) URI.create(url).toURL().openConnection();
            connection.setConnectTimeout(10_000);
            connection.setReadTimeout(60_000);
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json");
            if (!bearerToken.isBlank()) {
                connection.setRequestProperty("Authorization", "Bearer " + bearerToken);
            }
            connection.setFixedLengthStreamingMode(requestBody.length);
            connection.setDoOutput(true);
            try {
                connection.getOutputStream().write(requestBody);
                int status = connection.getResponseCode();
                var responseStream = status >= 200 && status < 300
                        ? connection.getInputStream()
                        : connection.getErrorStream();
                byte[] responseBody = responseStream == null ? new byte[0] : responseStream.readAllBytes();
                String response = new String(responseBody, StandardCharsets.UTF_8);
                if (status >= 200 && status < 300) {
                    return MAPPER.readTree(response);
                }
                if ((status == 429 || status >= 500) && attempt < maxAttempts) {
                    Thread.sleep(1_000L << (attempt - 1));
                    continue;
                }
                throw new IllegalStateException("HTTP " + status + " from " + url + ": " + response);
            } catch (IOException exception) {
                if (attempt >= maxAttempts) {
                    throw exception;
                }
                Thread.sleep(1_000L << (attempt - 1));
            } finally {
                connection.disconnect();
            }
        }
        throw new IllegalStateException("embedding request retry loop exhausted");
    }

    private static List<RankedDocument> parseHits(JsonNode response) {
        List<RankedDocument> documents = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        for (JsonNode hit : response.path("hits").path("hits")) {
            RankedDocument document = parseHit(hit);
            if (document == null || !seen.add(document.docId())) {
                continue;
            }
            documents.add(document);
        }
        return documents;
    }

    private static RankedDocument parseHit(JsonNode hit) {
        JsonNode source = hit.path("_source");
        String docId = source.path("benchmarkDocId").asText();
        if (docId.isBlank()) {
            return null;
        }
        return new RankedDocument(
                docId,
                hit.path("_id").asText(),
                source.path("chunkId").asInt(),
                source.path("sourceType").asText(),
                source.path("sourcePath").asText(),
                source.path("title").asText(),
                source.path("textContent").asText(),
                source.path("classification").asText(),
                source.path("documentVersion").asText(),
                source.path("documentHash").asText(),
                source.path("sourceUpdatedAt").asText(),
                source.path("contentHash").asText(),
                source.path("indexedAt").asText(),
                hit.path("_score").asDouble());
    }

    private static List<Question> loadQuestions(Path path) throws Exception {
        JsonNode payload = MAPPER.readTree(path.toFile());
        JsonNode rows = payload.isArray() ? payload : payload.path("questions");
        List<Question> questions = new ArrayList<>();
        for (JsonNode row : rows) {
            questions.add(new Question(
                    row.path("id").asText(),
                    row.path("question").asText(),
                    strings(row.path("expected_doc_ids")),
                    strings(row.path("source_types")),
                    row.path("question_type").asText("unknown"),
                    row.path("gold_answer").asText(""),
                    strings(row.path("answer_facts"))));
        }
        return questions;
    }

    private static List<String> strings(JsonNode node) {
        List<String> values = new ArrayList<>();
        if (node.isTextual()) {
            values.add(node.asText());
        } else if (node.isArray()) {
            node.forEach(value -> values.add(value.asText()));
        }
        return values;
    }

    private static JsonNode loadIndexMetadata(Config config) throws Exception {
        JsonHttpClient http = new JsonHttpClient();
        JsonNode indexResponse = http.requireJson(
                "GET",
                stripTrailingSlash(config.esUrl()) + "/" + config.index(),
                null,
                "",
                2,
                Duration.ofSeconds(30));
        JsonNode indexNode = indexResponse.path(config.index());
        if (indexNode.isMissingNode() && indexResponse.fields().hasNext()) {
            indexNode = indexResponse.fields().next().getValue();
        }
        JsonNode mappings = indexNode.path("mappings");
        validateIndexMetadata(config, mappings);
        JsonNode properties = mappings.path("properties");
        JsonNode vector = properties.path("vector");
        int vectorDimension = vector.path("dims").asInt(vector.path("dimension").asInt(-1));
        JsonNode indexSettings = indexNode.path("settings").path("index");
        JsonNode bm25 = indexSettings.path("similarity").path("enterprise_bm25");
        ObjectNode metadata = MAPPER.createObjectNode();
        metadata.put("index", config.index());
        metadata.put("engine", config.engine());
        metadata.put("mapping_sha256", RunManifest.sha256Json(mappings));
        metadata.put("vector_dimension", vectorDimension);
        metadata.put("vector_similarity", vector.path("similarity").asText(""));
        metadata.put("text_analyzer", properties.path("textContent").path("analyzer").asText(""));
        metadata.put("bm25_k1", bm25.path("k1").asText(""));
        metadata.put("bm25_b", bm25.path("b").asText(""));
        metadata.put("number_of_shards", indexSettings.path("number_of_shards").asText(""));
        metadata.put("number_of_replicas", indexSettings.path("number_of_replicas").asText(""));
        metadata.set("mapping_meta", mappings.path("_meta").deepCopy());
        return metadata;
    }

    static void validateIndexMetadata(Config config, JsonNode mappings) {
        JsonNode properties = mappings.path("properties");
        JsonNode vector = properties.path("vector");
        int vectorDimension = vector.path("dims").asInt(vector.path("dimension").asInt(-1));
        if (vectorDimension != config.embeddingDimension()) {
            throw new IllegalStateException(
                    "index vector dimension mismatch: config=" + config.embeddingDimension()
                            + ", mapping=" + vectorDimension);
        }
        if (!config.evidenceEnabled()) {
            return;
        }
        for (String field : List.of("documentVersion", "documentHash", "contentHash")) {
            if (!"keyword".equals(properties.path(field).path("type").asText())) {
                throw new IllegalStateException(
                        "EvidenceBuilder requires index field " + field + " mapped as keyword");
            }
        }
        if (!"date".equals(properties.path("sourceUpdatedAt").path("type").asText())) {
            throw new IllegalStateException(
                    "EvidenceBuilder requires index field sourceUpdatedAt mapped as date");
        }
    }

    private static void createParent(Path path) throws IOException {
        Path parent = path.toAbsolutePath().normalize().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
    }

    private static String stripTrailingSlash(String value) {
        return value.endsWith("/") ? value.substring(0, value.length() - 1) : value;
    }

    static Map<String, Object> summarize(Config config, List<Question> questions, List<Result> results) {
        if (questions.size() != results.size()) {
            throw new IllegalArgumentException("questions and results must have the same size");
        }
        List<Integer> allIndexes = new ArrayList<>(questions.size());
        for (int index = 0; index < questions.size(); index++) {
            allIndexes.add(index);
        }

        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("run_id", config.runId());
        summary.put("experiment_name", config.experiment().name());
        summary.put("created_at", OffsetDateTime.now().toString());
        summary.put("runtime", "java");
        summary.put("engine", config.engine());
        summary.put("index", config.index());
        summary.put("embedding_model", config.embeddingModel());
        summary.put("embedding_dimension", config.embeddingDimension());
        summary.put("embedding_query_instruction", config.embeddingQueryInstruction());
        summary.put("embedding_api_format", config.embeddingApiFormat());
        summary.put("experiment_metadata", MAPPER.convertValue(config.experiment().metadata(), Map.class));
        summary.put("retrieval", "source_acl_" + config.retrievalMode());
        summary.put("retriever_k", config.retrieverK());
        summary.put("rrf_k", config.rrfK());
        summary.put("dense_weight", config.denseWeight());
        summary.put("bm25_weight", config.bm25Weight());
        summary.put("keyword_bm25_enabled", config.keywordBm25Enabled());
        summary.put("keyword_bm25_weight", config.keywordBm25Weight());
        summary.put("english_bm25_enabled", config.englishBm25Enabled());
        summary.put("english_bm25_weight", config.englishBm25Weight());
        summary.put("top_k_documents", config.topK());
        summary.put("evidence_enabled", config.evidenceEnabled());
        summary.put("evidence_top_documents", config.evidenceTopDocuments());
        summary.put("evidence_candidate_chunks_per_document", config.evidenceCandidateChunksPerDocument());
        summary.put("evidence_chunks_per_document", config.evidenceChunksPerDocument());
        summary.put("evidence_token_budget", config.evidenceTokenBudget());
        summary.put("evidence_per_document_token_budget", config.evidencePerDocumentTokenBudget());
        summary.put("evidence_max_chunk_tokens", config.evidenceMaxChunkTokens());
        summary.put("document_ranking_changed_by_evidence_count", 0);
        summary.put("document_hit_to_miss_count", 0);
        summary.put("document_miss_to_hit_count", 0);
        summary.putAll(evaluateGroup(config, questions, results, allIndexes));
        summary.put("by_question_type", groupedMetrics(config, questions, results, false));
        summary.put("by_source_type", groupedMetrics(config, questions, results, true));
        return summary;
    }

    private static Map<String, Object> evaluateGroup(
            Config config,
            List<Question> questions,
            List<Result> results,
            List<Integer> indexes) {
        int evaluable = 0;
        Map<Integer, Integer> hits = new LinkedHashMap<>();
        for (int k : List.of(1, 5, 10, 20)) {
            hits.put(k, 0);
        }
        double reciprocalSum = 0.0d;
        int allGoldHit10 = 0;
        int noGoldDocCount = 0;
        int sourceFilterViolationCount = 0;
        int evidenceSourceFilterViolationCount = 0;
        int evidenceConflictCaseCount = 0;
        List<Double> latencies = new ArrayList<>();
        List<Double> retrievalLatencies = new ArrayList<>();
        List<Double> evidenceLatencies = new ArrayList<>();
        List<Double> evidenceFactRecalls = new ArrayList<>();
        List<Double> evidenceFactCoverage = new ArrayList<>();
        List<Double> evidenceGoldAnswerRecalls = new ArrayList<>();
        List<Integer> documentContextTokens = new ArrayList<>();
        List<Integer> evidenceTokens = new ArrayList<>();
        List<Integer> contextTokens = new ArrayList<>();
        List<Integer> evidenceSpanCounts = new ArrayList<>();
        List<Integer> evidenceCandidateChunkCounts = new ArrayList<>();

        for (int index : indexes) {
            Question question = questions.get(index);
            Result result = results.get(index);
            latencies.add(result.latencyMs());
            retrievalLatencies.add(result.retrievalLatencyMs());
            evidenceLatencies.add(result.evidenceLatencyMs());
            int documentTokens = result.documents().stream()
                    .limit(10)
                    .mapToInt(document -> tokenCount(document.title() + "\n" + document.text()))
                    .sum();
            documentContextTokens.add(documentTokens);
            evidenceTokens.add(result.evidence().tokenCount());
            contextTokens.add(config.evidenceEnabled() ? result.evidence().tokenCount() : documentTokens);
            evidenceSpanCounts.add(result.evidence().spans().size());
            evidenceCandidateChunkCounts.add(result.evidence().candidateChunks());
            if (!result.evidence().conflicts().isEmpty()) {
                evidenceConflictCaseCount++;
            }
            if (result.evidenceScores().factTokenRecall() != null) {
                evidenceFactRecalls.add(result.evidenceScores().factTokenRecall());
            }
            if (result.evidenceScores().factCoverage() != null) {
                evidenceFactCoverage.add(result.evidenceScores().factCoverage());
            }
            if (result.evidenceScores().goldAnswerTokenRecall() != null) {
                evidenceGoldAnswerRecalls.add(result.evidenceScores().goldAnswerTokenRecall());
            }
            if (!question.sourceTypes().isEmpty()
                    && result.documents().stream().anyMatch(
                            document -> !question.sourceTypes().contains(document.sourceType()))) {
                sourceFilterViolationCount++;
            }
            if (!question.sourceTypes().isEmpty()
                    && result.evidence().spans().stream().anyMatch(
                            span -> !question.sourceTypes().contains(span.sourceType()))) {
                evidenceSourceFilterViolationCount++;
            }
            if (question.expectedDocIds().isEmpty()) {
                noGoldDocCount++;
                continue;
            }
            evaluable++;
            List<String> ranking = result.documents().stream().map(RankedDocument::docId).toList();
            Set<String> expected = new HashSet<>(question.expectedDocIds());
            int firstRank = firstRank(ranking, expected);
            for (int k : hits.keySet()) {
                if (firstRank > 0 && firstRank <= k) {
                    hits.put(k, hits.get(k) + 1);
                }
            }
            if (firstRank > 0 && firstRank <= 10) {
                reciprocalSum += 1.0d / firstRank;
            }
            if (new HashSet<>(ranking.subList(0, Math.min(10, ranking.size()))).containsAll(expected)) {
                allGoldHit10++;
            }
        }

        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("questions_total", indexes.size());
        summary.put("questions_evaluable", evaluable);
        for (int k : hits.keySet()) {
            summary.put("hit@" + k, ratio(hits.get(k), evaluable));
        }
        summary.put("mrr@10", evaluable == 0 ? null : reciprocalSum / evaluable);
        summary.put("answer_contains@10", null);
        summary.put("expected_doc_hit@10", ratio(hits.get(10), evaluable));
        summary.put("all_expected_docs_hit@10", ratio(allGoldHit10, evaluable));
        summary.put("no_gold_doc_count", noGoldDocCount);
        summary.put("invalid_extra_docs_count", 0);
        summary.put("source_filter_violation_count", sourceFilterViolationCount);
        summary.put("evidence_source_filter_violation_count", evidenceSourceFilterViolationCount);
        summary.put("evidence_conflict_case_count", evidenceConflictCaseCount);
        summary.put("evidence_fact_token_recall_avg", average(evidenceFactRecalls));
        summary.put("evidence_fact_coverage_avg", average(evidenceFactCoverage));
        summary.put("evidence_gold_answer_token_recall_avg", average(evidenceGoldAnswerRecalls));
        summary.put("avg_latency_ms", averageOrZero(latencies));
        summary.put("p95_latency_ms", percentile95(latencies));
        summary.put("avg_retrieval_latency_ms", averageOrZero(retrievalLatencies));
        summary.put("p95_retrieval_latency_ms", percentile95(retrievalLatencies));
        summary.put("avg_evidence_latency_ms", averageOrZero(evidenceLatencies));
        summary.put("p95_evidence_latency_ms", percentile95(evidenceLatencies));
        summary.put("avg_document_context_tokens", averageInts(documentContextTokens));
        summary.put("avg_evidence_tokens", averageInts(evidenceTokens));
        summary.put("avg_context_tokens", averageInts(contextTokens));
        summary.put("avg_evidence_spans", averageInts(evidenceSpanCounts));
        summary.put("avg_evidence_candidate_chunks", averageInts(evidenceCandidateChunkCounts));
        return summary;
    }

    private static Map<String, Object> groupedMetrics(
            Config config,
            List<Question> questions,
            List<Result> results,
            boolean bySourceType) {
        Map<String, List<Integer>> grouped = new java.util.TreeMap<>();
        for (int index = 0; index < questions.size(); index++) {
            Question question = questions.get(index);
            List<String> groups = bySourceType
                    ? (question.sourceTypes().isEmpty() ? List.of("unknown") : question.sourceTypes())
                    : List.of(question.questionType());
            for (String group : groups) {
                grouped.computeIfAbsent(group, ignored -> new ArrayList<>()).add(index);
            }
        }
        Map<String, Object> metrics = new LinkedHashMap<>();
        grouped.forEach((group, indexes) -> metrics.put(group, evaluateGroup(config, questions, results, indexes)));
        return metrics;
    }

    private static Double average(List<Double> values) {
        return values.isEmpty()
                ? null
                : values.stream().mapToDouble(Double::doubleValue).average().orElse(0.0d);
    }

    private static double averageOrZero(List<Double> values) {
        Double value = average(values);
        return value == null ? 0.0d : value;
    }

    private static double averageInts(List<Integer> values) {
        return values.stream().mapToInt(Integer::intValue).average().orElse(0.0d);
    }

    private static double percentile95(List<Double> values) {
        if (values.isEmpty()) {
            return 0.0d;
        }
        List<Double> ordered = values.stream().sorted().toList();
        return ordered.get(Math.max(0, (int) Math.ceil(ordered.size() * 0.95d) - 1));
    }

    private static Double ratio(int count, int total) {
        return total == 0 ? null : count / (double) total;
    }

    private static int tokenCount(String text) {
        int count = 0;
        var matcher = TOKEN_PATTERN.matcher(text.toLowerCase());
        while (matcher.find()) {
            count++;
        }
        return count;
    }

    private static int firstRank(List<String> ranking, Set<String> expected) {
        for (int index = 0; index < ranking.size(); index++) {
            if (expected.contains(ranking.get(index))) {
                return index + 1;
            }
        }
        return -1;
    }

    private static double elapsedMs(long started) {
        return (System.nanoTime() - started) / 1_000_000.0d;
    }

    record Question(
            String id,
            String text,
            List<String> expectedDocIds,
            List<String> sourceTypes,
            String questionType,
            String goldAnswer,
            List<String> answerFacts) {
    }

    record RankedDocument(
            String docId,
            String chunkEsId,
            int chunkId,
            String sourceType,
            String sourcePath,
            String title,
            String text,
            String classification,
            String documentVersion,
            String documentHash,
            String sourceUpdatedAt,
            String contentHash,
            String indexedAt,
            double score) {

        RankedDocument withScore(double newScore) {
            return new RankedDocument(
                    docId,
                    chunkEsId,
                    chunkId,
                    sourceType,
                    sourcePath,
                    title,
                    text,
                    classification,
                    documentVersion,
                    documentHash,
                    sourceUpdatedAt,
                    contentHash,
                    indexedAt,
                    newScore);
        }
    }

    record RouteRanking(String name, double weight, List<RankedDocument> documents) {
    }

    record RouteEvidence(EvidenceBuilder.RouteSignal signal, RankedDocument chunk) {
    }

    record FusionResult(
            List<RankedDocument> documents,
            Map<String, List<RouteEvidence>> routeEvidenceByDoc) {
    }

    record EvidenceScores(
            Double factTokenRecall,
            Double factCoverage,
            Double goldAnswerTokenRecall) {
    }

    record Result(
            List<RankedDocument> documents,
            Map<String, List<RouteEvidence>> routeEvidenceByDoc,
            EvidenceBuilder.EvidenceBundle evidence,
            EvidenceScores evidenceScores,
            double embeddingLatencyMs,
            double denseLatencyMs,
            double bm25LatencyMs,
            double keywordBm25LatencyMs,
            double englishBm25LatencyMs,
            double fusionLatencyMs,
            double evidenceLatencyMs,
            double retrievalLatencyMs,
            double latencyMs) {

        Result {
            documents = List.copyOf(documents);
            routeEvidenceByDoc = Map.copyOf(routeEvidenceByDoc);
        }

        Map<String, Object> toDetails(Question question) {
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("question_id", question.id());
            details.put("question", question.text());
            details.put("question_type", question.questionType());
            details.put("source_types", question.sourceTypes());
            details.put("expected_doc_ids", question.expectedDocIds());
            details.put("gold_answer", question.goldAnswer());
            details.put("answer_facts", question.answerFacts());
            details.put("evaluable", !question.expectedDocIds().isEmpty());
            List<Map<String, Object>> ranking = new ArrayList<>();
            for (int index = 0; index < documents.size(); index++) {
                RankedDocument document = documents.get(index);
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("rank", index + 1);
                row.put("doc_id", document.docId());
                row.put("chunk_id", document.chunkId());
                row.put("chunk_es_id", document.chunkEsId());
                row.put("source_type", document.sourceType());
                row.put("source_path", document.sourcePath());
                row.put("title", document.title());
                row.put("document_version", document.documentVersion());
                row.put("document_hash", document.documentHash());
                row.put("score", document.score());
                row.put(
                        "route_contributions",
                        routeEvidenceByDoc.getOrDefault(document.docId(), List.of()).stream()
                                .map(RouteEvidence::signal)
                                .toList());
                ranking.add(row);
            }
            details.put("ranked_documents", ranking);
            details.put("evidence", evidence);
            details.put("evidence_fact_token_recall", evidenceScores.factTokenRecall());
            details.put("evidence_fact_coverage", evidenceScores.factCoverage());
            details.put("evidence_gold_answer_token_recall", evidenceScores.goldAnswerTokenRecall());
            details.put("embedding_latency_ms", embeddingLatencyMs);
            details.put("dense_latency_ms", denseLatencyMs);
            details.put("bm25_latency_ms", bm25LatencyMs);
            details.put("keyword_bm25_latency_ms", keywordBm25LatencyMs);
            details.put("english_bm25_latency_ms", englishBm25LatencyMs);
            details.put("fusion_latency_ms", fusionLatencyMs);
            details.put("evidence_latency_ms", evidenceLatencyMs);
            details.put("retrieval_latency_ms", retrievalLatencyMs);
            details.put("latency_ms", latencyMs);
            return details;
        }

        Map<String, Object> toEvidenceRow(Config config, Question question) {
            Set<String> expected = new HashSet<>(question.expectedDocIds());
            List<String> topTen = documents.stream().limit(10).map(RankedDocument::docId).toList();
            List<Map<String, Object>> contexts = new ArrayList<>();
            for (EvidenceBuilder.EvidenceSpan span : evidence.spans()) {
                Map<String, Object> context = new LinkedHashMap<>();
                context.put("citation_id", span.citationId());
                context.put("rank", span.rank());
                context.put("document_rank", span.documentRank());
                context.put("doc_id", span.docId());
                context.put("chunk_es_id", span.chunkEsId());
                context.put("chunk_id", span.chunkId());
                context.put("title", span.title());
                context.put("source_type", span.sourceType());
                context.put("source_path", span.sourcePath());
                context.put("classification", span.classification());
                context.put("document_version", span.documentVersion());
                context.put("document_hash", span.documentHash());
                context.put("source_updated_at", span.sourceUpdatedAt());
                context.put("content_hash", span.contentHash());
                context.put("text", span.text());
                context.put("token_count", span.tokenCount());
                context.put("evidence_score", span.selectionScore());
                context.put("query_coverage", span.queryCoverage());
                context.put("route_signals", span.routeSignals());
                context.put("conflict_group", span.conflictGroup());
                contexts.add(context);
            }

            Map<String, Object> row = new LinkedHashMap<>();
            row.put("qid", question.id());
            row.put("question", question.text());
            row.put("question_type", question.questionType());
            row.put("source_types", question.sourceTypes());
            row.put("user_id", question.sourceTypes().size() == 1
                    ? "source_" + question.sourceTypes().get(0)
                    : "benchmark_user");
            row.put("tenant_id", "tenant_redwood");
            row.put("retrieval_candidate", config.runId());
            row.put("context_mode", "java_evidence_builder");
            row.put("expected_doc_ids", question.expectedDocIds());
            row.put("expected_accessible_doc_ids", question.expectedDocIds());
            row.put("gold_answer", question.goldAnswer());
            row.put("answer_facts", question.answerFacts());
            row.put("is_evaluable", !expected.isEmpty());
            row.put("retrieval_hit_at_10", !expected.isEmpty() && topTen.stream().anyMatch(expected::contains));
            row.put("all_expected_docs_hit_at_10", !expected.isEmpty() && new HashSet<>(topTen).containsAll(expected));
            row.put("ranked_doc_ids", topTen);
            row.put("contexts", contexts);
            row.put("evidence_conflicts", evidence.conflicts());
            row.put("evidence_token_count", evidence.tokenCount());
            row.put("evidence_fact_token_recall", evidenceScores.factTokenRecall());
            row.put("evidence_fact_coverage", evidenceScores.factCoverage());
            row.put("evidence_gold_answer_token_recall", evidenceScores.goldAnswerTokenRecall());
            return row;
        }
    }

    record Config(
            ExperimentConfig.Snapshot experiment,
            String runId,
            Path questions,
            Path output,
            Path detailsOutput,
            Path manifestOutput,
            Path evidenceOutput,
            String esUrl,
            String embeddingUrl,
            String embeddingModel,
            int embeddingDimension,
            String embeddingQueryInstruction,
            String embeddingApiFormat,
            String embeddingApiKey,
            String engine,
            String index,
            String retrievalMode,
            int retrieverK,
            int denseChunkCandidates,
            int denseNumCandidates,
            int rrfK,
            double denseWeight,
            double bm25Weight,
            boolean keywordBm25Enabled,
            double keywordBm25Weight,
            boolean englishBm25Enabled,
            double englishBm25Weight,
            int topK,
            boolean evidenceEnabled,
            int evidenceTopDocuments,
            int evidenceCandidateChunksPerDocument,
            int evidenceChunksPerDocument,
            int evidenceTokenBudget,
            int evidencePerDocumentTokenBudget,
            int evidenceMaxChunkTokens,
            double evidenceRedundancyPenalty,
            int progressEvery) {

        Config {
            if (runId == null || runId.isBlank()) {
                throw new IllegalArgumentException("--run-id must not be blank");
            }
            validateExperimentMetadata(experiment, embeddingModel, embeddingDimension);
            evidenceTopDocuments = Math.min(evidenceTopDocuments, topK);
            List<Path> protectedPaths = new ArrayList<>(List.of(
                    questions,
                    output,
                    detailsOutput,
                    manifestOutput));
            if (evidenceEnabled) {
                protectedPaths.add(evidenceOutput);
            }
            if (new HashSet<>(protectedPaths).size() != protectedPaths.size()) {
                throw new IllegalArgumentException(
                        "questions, summary, details, manifest, and enabled evidence outputs must use distinct paths");
            }
            if ("elasticsearch".equals(engine)
                    && !"bm25".equals(retrievalMode)
                    && denseNumCandidates < denseChunkCandidates) {
                throw new IllegalArgumentException(
                        "--dense-num-candidates must be >= --dense-chunk-candidates for Elasticsearch");
            }
        }

        private static final Set<String> KNOWN_ARGUMENTS = Set.of(
                "run-id",
                "questions",
                "output",
                "details-output",
                "manifest-output",
                "evidence-output",
                "es-url",
                "embedding-url",
                "embedding-model",
                "embedding-dimension",
                "embedding-query-instruction",
                "embedding-api-format",
                "embedding-api-key",
                "embedding-api-key-env",
                "engine",
                "index",
                "retrieval-mode",
                "retriever-k",
                "dense-chunk-candidates",
                "dense-num-candidates",
                "rrf-k",
                "dense-weight",
                "bm25-weight",
                "keyword-bm25-enabled",
                "keyword-bm25-weight",
                "english-bm25-enabled",
                "english-bm25-weight",
                "top-k",
                "evidence-enabled",
                "evidence-top-documents",
                "evidence-candidate-chunks-per-document",
                "evidence-chunks-per-document",
                "evidence-token-budget",
                "evidence-per-document-token-budget",
                "evidence-max-chunk-tokens",
                "evidence-redundancy-penalty",
                "progress-every");

        static Config parse(String[] args) {
            ExperimentConfig.Snapshot experiment;
            try {
                experiment = ExperimentConfig.load(args);
            } catch (IOException exception) {
                throw new IllegalArgumentException("failed to load experiment config", exception);
            }
            Arguments values = experiment.arguments();
            Set<String> unknownArguments = new java.util.TreeSet<>(values.asMap().keySet());
            unknownArguments.removeAll(KNOWN_ARGUMENTS);
            if (!unknownArguments.isEmpty()) {
                throw new IllegalArgumentException("unknown evaluate arguments: " + unknownArguments);
            }
            Path questions = normalized(values.requiredPath("questions"));
            Path output = normalized(values.requiredPath("output"));
            Path detailsOutput = normalized(values.requiredPath("details-output"));
            Path manifestOutput = values.contains("manifest-output")
                    ? normalized(Path.of(values.required("manifest-output")))
                    : derivedOutput(output, ".manifest.json");
            Path evidenceOutput = values.contains("evidence-output")
                    ? normalized(Path.of(values.required("evidence-output")))
                    : derivedOutput(detailsOutput, ".evidence.jsonl");
            String defaultRunId = "inline-cli".equals(experiment.name())
                    ? "paismart_java_enterpriserag_source_acl_" + values.string("retrieval-mode", "hybrid")
                    : experiment.name();
            String apiKeyEnvironment = values.string("embedding-api-key-env", "DASHSCOPE_API_KEY");
            boolean evidenceEnabled = values.bool("evidence-enabled", false);
            int evidenceTokenBudget = values.nonNegativeInt("evidence-token-budget", 6000);
            if (evidenceEnabled && evidenceTokenBudget == 0) {
                throw new IllegalArgumentException("--evidence-token-budget must be positive when evidence is enabled");
            }

            return new Config(
                    experiment,
                    values.string("run-id", defaultRunId),
                    questions,
                    output,
                    detailsOutput,
                    manifestOutput,
                    evidenceOutput,
                    values.string("es-url", "http://127.0.0.1:19200"),
                    values.string("embedding-url", "http://127.0.0.1:18080/v1/embeddings"),
                    values.string("embedding-model", "intfloat/multilingual-e5-small"),
                    values.positiveInt("embedding-dimension", 384),
                    values.string("embedding-query-instruction", ""),
                    embeddingApiFormat(values.string("embedding-api-format", "local")),
                    values.string(
                            "embedding-api-key",
                            System.getenv().getOrDefault(apiKeyEnvironment, "")),
                    engine(values.string("engine", "elasticsearch")),
                    values.string("index", "knowledge_base_benchmark_stream_v1"),
                    retrievalMode(values.string("retrieval-mode", "hybrid")),
                    values.positiveInt("retriever-k", 50),
                    values.positiveInt("dense-chunk-candidates", 500),
                    values.positiveInt("dense-num-candidates", 2500),
                    values.nonNegativeInt("rrf-k", 60),
                    values.positiveDouble("dense-weight", 0.5d),
                    values.positiveDouble("bm25-weight", 1.0d),
                    values.bool("keyword-bm25-enabled", false),
                    values.positiveDouble("keyword-bm25-weight", 1.25d),
                    values.bool("english-bm25-enabled", false),
                    values.positiveDouble("english-bm25-weight", 1.5d),
                    values.positiveInt("top-k", 50),
                    evidenceEnabled,
                    values.positiveInt("evidence-top-documents", 10),
                    values.positiveInt("evidence-candidate-chunks-per-document", 8),
                    values.positiveInt("evidence-chunks-per-document", 3),
                    evidenceTokenBudget,
                    values.positiveInt("evidence-per-document-token-budget", 1200),
                    values.positiveInt("evidence-max-chunk-tokens", 512),
                    values.nonNegativeDouble("evidence-redundancy-penalty", 0.35d),
                    values.nonNegativeInt("progress-every", 25));
        }

        EvidenceBuilder.Config evidenceConfig() {
            return new EvidenceBuilder.Config(
                    Math.min(evidenceTopDocuments, topK),
                    evidenceCandidateChunksPerDocument,
                    evidenceChunksPerDocument,
                    evidenceTokenBudget,
                    evidencePerDocumentTokenBudget,
                    evidenceMaxChunkTokens,
                    evidenceRedundancyPenalty);
        }

        private static void validateExperimentMetadata(
                ExperimentConfig.Snapshot experiment,
                String embeddingModel,
                int embeddingDimension) {
            JsonNode embedding = experiment.metadata().path("embedding");
            if (!embedding.isObject()) {
                return;
            }
            String metadataModel = embedding.path("model").asText("").trim();
            if (!metadataModel.isEmpty() && !metadataModel.equals(embeddingModel)) {
                throw new IllegalArgumentException(
                        "experiment metadata embedding.model does not match --embedding-model: "
                                + metadataModel + " != " + embeddingModel);
            }
            if (embedding.has("stored_dimension")) {
                int storedDimension = embedding.path("stored_dimension").asInt(-1);
                if (storedDimension != embeddingDimension) {
                    throw new IllegalArgumentException(
                            "experiment metadata embedding.stored_dimension does not match "
                                    + "--embedding-dimension: " + storedDimension + " != " + embeddingDimension);
                }
            }
            if (embedding.has("native_dimension")
                    && embedding.path("native_dimension").asInt(-1) <= 0) {
                throw new IllegalArgumentException(
                        "experiment metadata embedding.native_dimension must be positive");
            }
        }

        private static Path normalized(Path path) {
            return path.toAbsolutePath().normalize();
        }

        private static Path derivedOutput(Path source, String suffix) {
            Path filename = source.getFileName();
            String name = filename == null ? "run" : filename.toString();
            int dot = name.lastIndexOf('.');
            String base = dot > 0 ? name.substring(0, dot) : name;
            Path sibling = Path.of(base + suffix);
            Path parent = source.getParent();
            return parent == null ? sibling : parent.resolve(sibling);
        }

        private static String retrievalMode(String value) {
            if (!Set.of("dense", "bm25", "hybrid").contains(value)) {
                throw new IllegalArgumentException("--retrieval-mode must be dense, bm25, or hybrid");
            }
            return value;
        }

        private static String embeddingApiFormat(String value) {
            if (!Set.of("local", "openai", "dashscope").contains(value)) {
                throw new IllegalArgumentException(
                        "--embedding-api-format must be local, openai, or dashscope");
            }
            return value;
        }

        private static String engine(String value) {
            if (!Set.of("elasticsearch", "opensearch").contains(value)) {
                throw new IllegalArgumentException("--engine must be elasticsearch or opensearch");
            }
            return value;
        }
    }
}
