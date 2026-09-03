package com.yizhaoqi.smartpai.service;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class EvidenceBuilderTest {

    private final EvidenceBuilder builder = new EvidenceBuilder();

    @Test
    void selectsComplementaryChunksAndPreservesRouteSignals() {
        EvidenceBuilder.RouteSignal dense = signal("dense", 1, 0.75d, "doc-1:00001");
        EvidenceBuilder.RouteSignal bm25 = signal("bm25_original", 1, 0.50d, "doc-1:00001");
        EvidenceBuilder.DocumentCandidate document = new EvidenceBuilder.DocumentCandidate(
                "doc-1",
                1,
                dense.rrfContribution() + bm25.rrfContribution(),
                List.of(dense, bm25),
                List.of(
                        chunk(
                                "doc-1",
                                1,
                                "The default max_file_size for multipart uploads is 10 MiB per file.",
                                8.0d,
                                List.of(dense, bm25),
                                "source:upload"),
                        chunk(
                                "doc-1",
                                2,
                                "The default max_file_size for multipart uploads is 10 MiB per file and is configurable.",
                                7.8d,
                                List.of(),
                                "source:upload"),
                        chunk(
                                "doc-1",
                                3,
                                "The default max_total_request_size is 50 MiB for the complete request.",
                                7.5d,
                                List.of(),
                                "source:upload")));

        EvidenceBuilder.EvidenceBundle result = builder.build(
                "What are the default max_file_size and max_total_request_size for multipart uploads?",
                List.of(document),
                new EvidenceBuilder.Config(1, 2, 160, 160, 80, 0.60d));

        assertThat(result.spans()).hasSize(2);
        assertThat(result.spans()).extracting(EvidenceBuilder.EvidenceSpan::chunkId)
                .containsExactly(1, 3);
        assertThat(result.spans().get(0).routeSignals())
                .extracting(EvidenceBuilder.RouteSignal::route)
                .containsExactlyInAnyOrder("dense", "bm25_original");
        assertThat(result.tokenCount()).isLessThanOrEqualTo(160);
    }

    @Test
    void enforcesGlobalAndPerDocumentTokenBudgets() {
        String longText = "token ".repeat(120);
        EvidenceBuilder.DocumentCandidate first = document(
                "doc-1",
                1,
                "source:one",
                longText,
                "v1",
                "hash-1");
        EvidenceBuilder.DocumentCandidate second = document(
                "doc-2",
                2,
                "source:two",
                longText,
                "v1",
                "hash-2");

        EvidenceBuilder.EvidenceBundle result = builder.build(
                "token",
                List.of(first, second),
                new EvidenceBuilder.Config(2, 2, 50, 40, 40, 0.35d));

        assertThat(result.tokenCount()).isLessThanOrEqualTo(50);
        assertThat(result.spans()).isNotEmpty();
        assertThat(result.spans()).allSatisfy(span ->
                assertThat(span.tokenCount()).isLessThanOrEqualTo(40));
    }

    @Test
    void ignoresLowInformationQuestionWordsWhenCalculatingCoverage() {
        EvidenceBuilder.RouteSignal signal = signal("bm25_original", 1, 1.0d, "doc-stopwords:00001");
        EvidenceBuilder.DocumentCandidate document = new EvidenceBuilder.DocumentCandidate(
                "doc-stopwords",
                1,
                signal.rrfContribution(),
                List.of(signal),
                List.of(chunk(
                        "doc-stopwords",
                        1,
                        "What are the details?",
                        2.0d,
                        List.of(signal),
                        "source:stopwords")));

        EvidenceBuilder.EvidenceBundle result = builder.build(
                "What are the deployment limits?",
                List.of(document),
                new EvidenceBuilder.Config(1, 1, 80, 80, 80, 0.35d));

        assertThat(result.spans()).hasSize(1);
        assertThat(result.spans().get(0).queryCoverage()).isZero();
    }

    @Test
    void centersBudgetedEvidenceAroundQueryTermsNearTheChunkTail() {
        EvidenceBuilder.RouteSignal signal = signal("bm25_original", 1, 1.0d, "doc-tail:00001");
        String text = "filler ".repeat(40) + "criticalneedle";
        EvidenceBuilder.DocumentCandidate document = new EvidenceBuilder.DocumentCandidate(
                "doc-tail",
                1,
                signal.rrfContribution(),
                List.of(signal),
                List.of(chunk(
                        "doc-tail",
                        1,
                        text,
                        9.0d,
                        List.of(signal),
                        "source:tail")));

        EvidenceBuilder.EvidenceBundle result = builder.build(
                "criticalneedle",
                List.of(document),
                new EvidenceBuilder.Config(1, 1, 20, 20, 20, 0.35d));

        assertThat(result.spans()).hasSize(1);
        assertThat(result.spans().get(0).text()).contains("criticalneedle");
        assertThat(result.spans().get(0).text()).startsWith("… ");
        assertThat(result.spans().get(0).queryCoverage()).isEqualTo(1.0d);
        assertThat(result.spans().get(0).tokenCount()).isLessThanOrEqualTo(20);
    }

    @Test
    void marksConflictingVersionsFromTheSameSourceInsteadOfSilentlyMergingThem() {
        EvidenceBuilder.DocumentCandidate oldVersion = document(
                "doc-old",
                1,
                "confluence:access-policy",
                "Contractor access lasts 60 days.",
                "v1",
                "hash-old");
        EvidenceBuilder.DocumentCandidate newVersion = document(
                "doc-new",
                2,
                "confluence:access-policy",
                "Contractor access lasts 90 days.",
                "v2",
                "hash-new");

        EvidenceBuilder.EvidenceBundle result = builder.build(
                "How long does contractor access last?",
                List.of(oldVersion, newVersion),
                new EvidenceBuilder.Config(2, 1, 100, 60, 60, 0.35d));

        assertThat(result.conflicts()).hasSize(1);
        assertThat(result.conflicts().get(0).sourcePath()).isEqualTo("confluence:access-policy");
        assertThat(result.conflicts().get(0).documentVersions()).containsExactly("v1", "v2");
        assertThat(result.conflicts().get(0).resolution()).isEqualTo("preserve_and_mark");
        assertThat(result.spans()).allSatisfy(span ->
                assertThat(span.conflictGroup()).isEqualTo("source:confluence:access-policy"));
    }

    @Test
    void marksStaleChunksFromDifferentVersionsInsideTheSameDocument() {
        EvidenceBuilder.RouteSignal signal = signal("bm25_original", 1, 1.0d, "doc-stale:00001");
        EvidenceBuilder.DocumentCandidate document = new EvidenceBuilder.DocumentCandidate(
                "doc-stale",
                1,
                signal.rrfContribution(),
                List.of(signal),
                List.of(
                        new EvidenceBuilder.ChunkCandidate(
                                "doc-stale",
                                "doc-stale:00001",
                                1,
                                "confluence",
                                "confluence:access-policy",
                                "Policy",
                                "The old policy allowed 60 days.",
                                "internal",
                                "v1",
                                "hash-old",
                                "2026-07-01T00:00:00Z",
                                "chunk-old",
                                5.0d,
                                List.of(signal)),
                        new EvidenceBuilder.ChunkCandidate(
                                "doc-stale",
                                "doc-stale:00002",
                                2,
                                "confluence",
                                "confluence:access-policy",
                                "Policy",
                                "The current policy allows 90 days.",
                                "internal",
                                "v2",
                                "hash-new",
                                "2026-08-01T00:00:00Z",
                                "chunk-new",
                                5.0d,
                                List.of())));

        EvidenceBuilder.EvidenceBundle result = builder.build(
                "How many days does the access policy allow?",
                List.of(document),
                new EvidenceBuilder.Config(1, 2, 100, 100, 50, 0.35d));

        assertThat(result.conflicts()).hasSize(1);
        assertThat(result.conflicts().get(0).docIds()).containsExactly("doc-stale");
        assertThat(result.conflicts().get(0).documentVersions()).containsExactly("v1", "v2");
        assertThat(result.conflicts().get(0).documentHashes()).containsExactly("hash-new", "hash-old");
        assertThat(result.spans()).allSatisfy(span ->
                assertThat(span.conflictGroup()).isEqualTo("source:confluence:access-policy"));
    }

    private static EvidenceBuilder.DocumentCandidate document(
            String docId,
            int rank,
            String sourcePath,
            String text,
            String version,
            String hash) {
        EvidenceBuilder.RouteSignal signal = signal("bm25_original", rank, 1.0d, docId + ":00001");
        return new EvidenceBuilder.DocumentCandidate(
                docId,
                rank,
                signal.rrfContribution(),
                List.of(signal),
                List.of(new EvidenceBuilder.ChunkCandidate(
                        docId,
                        docId + ":00001",
                        1,
                        "confluence",
                        sourcePath,
                        "Policy",
                        text,
                        "internal",
                        version,
                        hash,
                        "2026-08-01T00:00:00Z",
                        "chunk-hash-" + docId,
                        4.0d,
                        List.of(signal))));
    }

    private static EvidenceBuilder.ChunkCandidate chunk(
            String docId,
            int chunkId,
            String text,
            double lexicalScore,
            List<EvidenceBuilder.RouteSignal> signals,
            String sourcePath) {
        return new EvidenceBuilder.ChunkCandidate(
                docId,
                docId + ":" + String.format("%05d", chunkId),
                chunkId,
                "github",
                sourcePath,
                "Multipart defaults",
                text,
                "internal",
                "v1",
                "doc-hash",
                "2026-08-01T00:00:00Z",
                "chunk-hash-" + chunkId,
                lexicalScore,
                signals);
    }

    private static EvidenceBuilder.RouteSignal signal(
            String route,
            int rank,
            double weight,
            String chunkEsId) {
        return new EvidenceBuilder.RouteSignal(
                route,
                rank,
                weight,
                10.0d,
                weight / (10.0d + rank),
                chunkEsId);
    }
}
