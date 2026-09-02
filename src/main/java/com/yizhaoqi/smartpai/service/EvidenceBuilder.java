package com.yizhaoqi.smartpai.service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Selects bounded, complementary chunk evidence without changing document order.
 *
 * <p>The builder receives the fused document ranking, representative chunks
 * retained from each retrieval route, and bounded lexical candidates fetched
 * from the top parent documents. It then performs deterministic MMR-like
 * selection under per-document and global token budgets.</p>
 */
public final class EvidenceBuilder {

    private static final Pattern TOKEN_PATTERN = Pattern.compile(
            "[a-z0-9_][a-z0-9_./:+#-]*|[\\u4e00-\\u9fff]",
            Pattern.CASE_INSENSITIVE);
    private static final int MIN_EVIDENCE_TOKENS = 24;

    public EvidenceBundle build(String query, List<DocumentCandidate> documents, Config config) {
        Config validated = config.validated();
        List<DocumentCandidate> topDocuments = documents.stream()
                .sorted(Comparator.comparingInt(DocumentCandidate::rank))
                .limit(validated.topDocuments())
                .toList();
        if (topDocuments.isEmpty() || validated.tokenBudget() == 0) {
            return new EvidenceBundle(List.of(), 0, List.of(), topDocuments.size(), 0);
        }

        Set<String> queryTokens = tokenSet(Bm25QueryRewriter.keywordQuery(query));
        List<Conflict> conflicts = detectConflicts(topDocuments);
        Set<String> conflictingSources = conflicts.stream()
                .map(Conflict::sourcePath)
                .collect(LinkedHashSet::new, Set::add, Set::addAll);

        Map<String, List<Selection>> selectionsByDocument = new LinkedHashMap<>();
        int candidateChunks = 0;
        for (DocumentCandidate document : topDocuments) {
            List<ChunkCandidate> merged = mergeChunks(document.chunks());
            candidateChunks += merged.size();
            selectionsByDocument.put(
                    document.docId(),
                    selectForDocument(queryTokens, document, merged, validated));
        }

        List<EvidenceSpan> spans = new ArrayList<>();
        Map<String, Set<String>> emittedQueryTokensByDocument = new LinkedHashMap<>();
        int remainingGlobalTokens = validated.tokenBudget();
        for (int round = 0; round < validated.chunksPerDocument() && remainingGlobalTokens > 0; round++) {
            for (DocumentCandidate document : topDocuments) {
                List<Selection> selections = selectionsByDocument.getOrDefault(document.docId(), List.of());
                if (round >= selections.size() || remainingGlobalTokens <= 0) {
                    continue;
                }
                Selection selection = selections.get(round);
                int allowed = Math.min(selection.tokenCount(), remainingGlobalTokens);
                boolean wouldCreateFragment = allowed < selection.tokenCount();
                if (wouldCreateFragment && allowed < MIN_EVIDENCE_TOKENS && !spans.isEmpty()) {
                    continue;
                }
                String text = selection.text();
                int tokens = selection.tokenCount();
                if (tokens > allowed) {
                    text = truncateToTokens(text, allowed);
                    tokens = tokenCount(text);
                }
                ChunkCandidate chunk = selection.chunk();
                Set<String> finalTokens = tokenSet(chunk.title() + "\n" + text);
                Set<String> finalMatchedQueryTokens = intersection(queryTokens, finalTokens);
                double finalQueryCoverage = ratio(finalMatchedQueryTokens.size(), queryTokens.size());
                Set<String> previouslyEmitted = emittedQueryTokensByDocument.computeIfAbsent(
                        chunk.docId(), ignored -> new HashSet<>());
                Set<String> finalNovelQueryTokens = new HashSet<>(finalMatchedQueryTokens);
                finalNovelQueryTokens.removeAll(previouslyEmitted);
                double finalNovelty = ratio(finalNovelQueryTokens.size(), queryTokens.size());
                previouslyEmitted.addAll(finalMatchedQueryTokens);
                String conflictGroup = conflictingSources.contains(chunk.sourcePath())
                        ? "source:" + chunk.sourcePath()
                        : "";
                spans.add(new EvidenceSpan(
                        "S" + (spans.size() + 1),
                        spans.size() + 1,
                        chunk.docId(),
                        document.rank(),
                        chunk.chunkEsId(),
                        chunk.chunkId(),
                        chunk.sourceType(),
                        chunk.sourcePath(),
                        chunk.title(),
                        chunk.classification(),
                        chunk.documentVersion(),
                        chunk.documentHash(),
                        chunk.sourceUpdatedAt(),
                        chunk.contentHash(),
                        text,
                        tokens,
                        selection.selectionScore(),
                        selection.baseScore(),
                        finalQueryCoverage,
                        selection.lexicalScore(),
                        selection.routeSupport(),
                        finalNovelty,
                        chunk.routeSignals(),
                        conflictGroup));
                remainingGlobalTokens -= tokens;
            }
        }

        int totalTokens = spans.stream().mapToInt(EvidenceSpan::tokenCount).sum();
        return new EvidenceBundle(
                List.copyOf(spans),
                totalTokens,
                conflicts,
                topDocuments.size(),
                candidateChunks);
    }

    private static List<Selection> selectForDocument(
            Set<String> queryTokens,
            DocumentCandidate document,
            List<ChunkCandidate> chunks,
            Config config) {
        if (chunks.isEmpty()) {
            return List.of();
        }
        double maxLexical = chunks.stream()
                .mapToDouble(chunk -> Math.max(0.0d, chunk.lexicalScore()))
                .max()
                .orElse(0.0d);
        List<CandidateState> candidates = new ArrayList<>();
        for (ChunkCandidate chunk : chunks) {
            int scoringTextLimit = Math.min(config.maxChunkTokens(), config.perDocumentTokenBudget());
            String scoringText = truncateToTokens(chunk.text(), scoringTextLimit);
            Set<String> tokens = tokenSet(chunk.title() + "\n" + scoringText);
            Set<String> matchedQueryTokens = intersection(queryTokens, tokens);
            double queryCoverage = ratio(matchedQueryTokens.size(), queryTokens.size());
            double lexical = maxLexical <= 0.0d
                    ? 0.0d
                    : Math.log1p(Math.max(0.0d, chunk.lexicalScore())) / Math.log1p(maxLexical);
            double chunkContribution = chunk.routeSignals().stream()
                    .mapToDouble(RouteSignal::rrfContribution)
                    .sum();
            double routeSupport = document.fusionScore() <= 0.0d
                    ? 0.0d
                    : Math.min(1.0d, chunkContribution / document.fusionScore());
            double documentPrior = 1.0d / (1.0d + Math.log1p(Math.max(0, document.rank() - 1)));
            double baseScore = 0.45d * queryCoverage
                    + 0.25d * lexical
                    + 0.20d * routeSupport
                    + 0.10d * documentPrior;
            candidates.add(new CandidateState(
                    chunk,
                    tokens,
                    matchedQueryTokens,
                    queryCoverage,
                    lexical,
                    routeSupport,
                    baseScore));
        }

        List<Selection> selected = new ArrayList<>();
        Set<String> selectedKeys = new HashSet<>();
        Set<String> coveredQueryTokens = new HashSet<>();
        int remainingDocumentTokens = config.perDocumentTokenBudget();
        while (selected.size() < config.chunksPerDocument()
                && remainingDocumentTokens > 0) {
            CandidateChoice best = null;
            for (CandidateState candidate : candidates) {
                String key = chunkKey(candidate.chunk());
                if (selectedKeys.contains(key)) {
                    continue;
                }
                double redundancy = selected.stream()
                        .mapToDouble(value -> jaccard(candidate.tokens(), value.tokens()))
                        .max()
                        .orElse(0.0d);
                Set<String> novelQueryTokens = new HashSet<>(candidate.matchedQueryTokens());
                novelQueryTokens.removeAll(coveredQueryTokens);
                double novelty = ratio(novelQueryTokens.size(), queryTokens.size());
                double neighborBonus = selected.stream().anyMatch(value ->
                        value.chunk().docId().equals(candidate.chunk().docId())
                                && Math.abs(value.chunk().chunkId() - candidate.chunk().chunkId()) == 1)
                        ? 0.05d
                        : 0.0d;
                double selectionScore = candidate.baseScore()
                        + 0.20d * novelty
                        + neighborBonus
                        - config.redundancyPenalty() * redundancy;
                CandidateChoice choice = new CandidateChoice(candidate, selectionScore, novelty);
                if (best == null || compare(choice, best) < 0) {
                    best = choice;
                }
            }
            if (best == null) {
                break;
            }

            CandidateState candidate = best.candidate();
            int originalTokens = tokenCount(candidate.chunk().text());
            int allowedTokens = Math.min(
                    Math.min(config.maxChunkTokens(), remainingDocumentTokens),
                    originalTokens);
            boolean wouldCreateFragment = allowedTokens < originalTokens;
            if (wouldCreateFragment && allowedTokens < MIN_EVIDENCE_TOKENS && !selected.isEmpty()) {
                break;
            }
            String text = truncateToTokens(candidate.chunk().text(), allowedTokens);
            int tokens = tokenCount(text);
            if (tokens == 0) {
                selectedKeys.add(chunkKey(candidate.chunk()));
                continue;
            }
            Set<String> selectedTokens = tokenSet(candidate.chunk().title() + "\n" + text);
            Set<String> selectedMatchedQueryTokens = intersection(queryTokens, selectedTokens);
            selected.add(new Selection(
                    candidate.chunk(),
                    selectedTokens,
                    text,
                    tokens,
                    best.selectionScore(),
                    candidate.baseScore(),
                    candidate.lexicalScore(),
                    candidate.routeSupport()));
            selectedKeys.add(chunkKey(candidate.chunk()));
            coveredQueryTokens.addAll(selectedMatchedQueryTokens);
            remainingDocumentTokens -= tokens;
        }
        return List.copyOf(selected);
    }

    private static int compare(CandidateChoice left, CandidateChoice right) {
        int score = Double.compare(right.selectionScore(), left.selectionScore());
        if (score != 0) {
            return score;
        }
        int routes = Integer.compare(
                right.candidate().chunk().routeSignals().size(),
                left.candidate().chunk().routeSignals().size());
        if (routes != 0) {
            return routes;
        }
        return Integer.compare(left.candidate().chunk().chunkId(), right.candidate().chunk().chunkId());
    }

    private static List<ChunkCandidate> mergeChunks(List<ChunkCandidate> chunks) {
        Map<String, MutableChunk> values = new LinkedHashMap<>();
        for (ChunkCandidate chunk : chunks) {
            String key = chunkKey(chunk);
            values.computeIfAbsent(key, ignored -> new MutableChunk(chunk)).merge(chunk);
        }
        return values.values().stream().map(MutableChunk::freeze).toList();
    }

    private static String chunkKey(ChunkCandidate chunk) {
        if (!chunk.chunkEsId().isBlank()) {
            return chunk.chunkEsId();
        }
        return chunk.docId() + ":" + chunk.chunkId();
    }

    private static List<Conflict> detectConflicts(List<DocumentCandidate> documents) {
        Map<String, List<DocumentCandidate>> bySource = new LinkedHashMap<>();
        for (DocumentCandidate document : documents) {
            String sourcePath = document.chunks().stream()
                    .map(ChunkCandidate::sourcePath)
                    .filter(value -> !value.isBlank())
                    .findFirst()
                    .orElse("");
            if (!sourcePath.isBlank()) {
                bySource.computeIfAbsent(sourcePath, ignored -> new ArrayList<>()).add(document);
            }
        }

        List<Conflict> conflicts = new ArrayList<>();
        bySource.forEach((sourcePath, candidates) -> {
            Set<String> docIds = new TreeSet<>();
            Set<String> versions = new TreeSet<>();
            Set<String> hashes = new TreeSet<>();
            candidates.forEach(document -> {
                docIds.add(document.docId());
                document.chunks().forEach(chunk -> {
                    if (!chunk.documentVersion().isBlank()) {
                        versions.add(chunk.documentVersion());
                    }
                    if (!chunk.documentHash().isBlank()) {
                        hashes.add(chunk.documentHash());
                    }
                });
            });
            if (docIds.size() > 1 && (versions.size() > 1 || hashes.size() > 1)) {
                conflicts.add(new Conflict(
                        sourcePath,
                        List.copyOf(docIds),
                        List.copyOf(versions),
                        List.copyOf(hashes),
                        "preserve_and_mark"));
            }
        });
        return List.copyOf(conflicts);
    }

    static int tokenCount(String text) {
        int count = 0;
        Matcher matcher = TOKEN_PATTERN.matcher(text == null ? "" : text.toLowerCase(Locale.ROOT));
        while (matcher.find()) {
            count++;
        }
        return count;
    }

    static String truncateToTokens(String text, int maxTokens) {
        if (text == null || text.isBlank() || maxTokens <= 0) {
            return "";
        }
        Matcher matcher = TOKEN_PATTERN.matcher(text);
        int count = 0;
        int end = 0;
        while (matcher.find()) {
            count++;
            end = matcher.end();
            if (count >= maxTokens) {
                break;
            }
        }
        if (count < maxTokens || end >= text.length()) {
            return text.trim();
        }
        return text.substring(0, end).trim() + " …";
    }

    private static Set<String> tokenSet(String text) {
        Set<String> tokens = new LinkedHashSet<>();
        Matcher matcher = TOKEN_PATTERN.matcher(text == null ? "" : text.toLowerCase(Locale.ROOT));
        while (matcher.find()) {
            String token = matcher.group();
            if (!token.isBlank()) {
                tokens.add(token);
            }
        }
        return tokens;
    }

    private static Set<String> intersection(Set<String> left, Set<String> right) {
        Set<String> values = new HashSet<>(left);
        values.retainAll(right);
        return values;
    }

    private static double jaccard(Set<String> left, Set<String> right) {
        if (left.isEmpty() && right.isEmpty()) {
            return 0.0d;
        }
        Set<String> intersection = new HashSet<>(left);
        intersection.retainAll(right);
        Set<String> union = new HashSet<>(left);
        union.addAll(right);
        return ratio(intersection.size(), union.size());
    }

    private static double ratio(int count, int total) {
        return total <= 0 ? 0.0d : count / (double) total;
    }

    public record Config(
            int topDocuments,
            int candidateChunksPerDocument,
            int chunksPerDocument,
            int tokenBudget,
            int perDocumentTokenBudget,
            int maxChunkTokens,
            double redundancyPenalty) {

        public Config validated() {
            if (topDocuments <= 0 || candidateChunksPerDocument <= 0 || chunksPerDocument <= 0) {
                throw new IllegalArgumentException("evidence document and chunk limits must be positive");
            }
            if (tokenBudget < 0 || perDocumentTokenBudget <= 0 || maxChunkTokens <= 0) {
                throw new IllegalArgumentException("evidence token budgets are invalid");
            }
            if (!Double.isFinite(redundancyPenalty) || redundancyPenalty < 0.0d) {
                throw new IllegalArgumentException("evidence redundancy penalty must not be negative");
            }
            return this;
        }
    }

    public record RouteSignal(
            String route,
            int rank,
            double weight,
            double rawScore,
            double rrfContribution,
            String chunkEsId) {

        public RouteSignal {
            route = value(route);
            chunkEsId = value(chunkEsId);
            if (rank <= 0 || !Double.isFinite(weight) || weight <= 0.0d
                    || !Double.isFinite(rawScore) || !Double.isFinite(rrfContribution)) {
                throw new IllegalArgumentException("invalid route signal");
            }
        }
    }

    public record ChunkCandidate(
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
            double lexicalScore,
            List<RouteSignal> routeSignals) {

        public ChunkCandidate {
            docId = value(docId);
            chunkEsId = value(chunkEsId);
            sourceType = value(sourceType);
            sourcePath = value(sourcePath);
            title = value(title);
            text = value(text);
            classification = value(classification);
            documentVersion = value(documentVersion);
            documentHash = value(documentHash);
            sourceUpdatedAt = value(sourceUpdatedAt);
            contentHash = value(contentHash);
            routeSignals = routeSignals == null ? List.of() : List.copyOf(routeSignals);
            if (docId.isBlank() || chunkId < 0 || !Double.isFinite(lexicalScore)) {
                throw new IllegalArgumentException("invalid evidence chunk candidate");
            }
        }
    }

    public record DocumentCandidate(
            String docId,
            int rank,
            double fusionScore,
            List<RouteSignal> routeSignals,
            List<ChunkCandidate> chunks) {

        public DocumentCandidate {
            docId = value(docId);
            routeSignals = routeSignals == null ? List.of() : List.copyOf(routeSignals);
            chunks = chunks == null ? List.of() : List.copyOf(chunks);
            if (docId.isBlank() || rank <= 0 || !Double.isFinite(fusionScore)) {
                throw new IllegalArgumentException("invalid evidence document candidate");
            }
        }
    }

    public record EvidenceSpan(
            String citationId,
            int rank,
            String docId,
            int documentRank,
            String chunkEsId,
            int chunkId,
            String sourceType,
            String sourcePath,
            String title,
            String classification,
            String documentVersion,
            String documentHash,
            String sourceUpdatedAt,
            String contentHash,
            String text,
            int tokenCount,
            double selectionScore,
            double baseScore,
            double queryCoverage,
            double lexicalScore,
            double routeSupport,
            double noveltyScore,
            List<RouteSignal> routeSignals,
            String conflictGroup) {

        public EvidenceSpan {
            routeSignals = routeSignals == null ? List.of() : List.copyOf(routeSignals);
        }
    }

    public record Conflict(
            String sourcePath,
            List<String> docIds,
            List<String> documentVersions,
            List<String> documentHashes,
            String resolution) {
    }

    public record EvidenceBundle(
            List<EvidenceSpan> spans,
            int tokenCount,
            List<Conflict> conflicts,
            int candidateDocuments,
            int candidateChunks) {

        public EvidenceBundle {
            spans = spans == null ? List.of() : List.copyOf(spans);
            conflicts = conflicts == null ? List.of() : List.copyOf(conflicts);
        }
    }

    private static String value(String text) {
        return text == null ? "" : text;
    }

    private record CandidateState(
            ChunkCandidate chunk,
            Set<String> tokens,
            Set<String> matchedQueryTokens,
            double queryCoverage,
            double lexicalScore,
            double routeSupport,
            double baseScore) {
    }

    private record CandidateChoice(
            CandidateState candidate,
            double selectionScore,
            double noveltyScore) {
    }

    private record Selection(
            ChunkCandidate chunk,
            Set<String> tokens,
            String text,
            int tokenCount,
            double selectionScore,
            double baseScore,
            double lexicalScore,
            double routeSupport) {
    }

    private static final class MutableChunk {
        private ChunkCandidate preferred;
        private double lexicalScore;
        private final Map<String, RouteSignal> routes = new LinkedHashMap<>();

        private MutableChunk(ChunkCandidate chunk) {
            this.preferred = chunk;
            this.lexicalScore = chunk.lexicalScore();
        }

        private void merge(ChunkCandidate chunk) {
            if (preferred.text().isBlank() && !chunk.text().isBlank()) {
                preferred = chunk;
            }
            lexicalScore = Math.max(lexicalScore, chunk.lexicalScore());
            for (RouteSignal signal : chunk.routeSignals()) {
                routes.merge(signal.route(), signal, (left, right) -> left.rank() <= right.rank() ? left : right);
            }
        }

        private ChunkCandidate freeze() {
            return new ChunkCandidate(
                    preferred.docId(),
                    preferred.chunkEsId(),
                    preferred.chunkId(),
                    preferred.sourceType(),
                    preferred.sourcePath(),
                    preferred.title(),
                    preferred.text(),
                    preferred.classification(),
                    preferred.documentVersion(),
                    preferred.documentHash(),
                    preferred.sourceUpdatedAt(),
                    preferred.contentHash(),
                    lexicalScore,
                    List.copyOf(routes.values()));
        }
    }
}
