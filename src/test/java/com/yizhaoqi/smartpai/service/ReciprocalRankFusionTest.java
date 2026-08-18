package com.yizhaoqi.smartpai.service;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ReciprocalRankFusionTest {

    @Test
    void promotesDocumentsFoundByBothRetrievers() {
        List<ReciprocalRankFusion.FusedResult<String>> fused = ReciprocalRankFusion.fuse(
                List.of(
                        List.of("dense-only", "shared", "tail"),
                        List.of("bm25-only", "shared", "other")),
                value -> value,
                60,
                4);

        assertThat(fused).extracting(ReciprocalRankFusion.FusedResult::value)
                .containsExactly("shared", "dense-only", "bm25-only", "tail");
        assertThat(fused.get(0).score()).isGreaterThan(fused.get(1).score());
    }

    @Test
    void usesStableFirstSeenOrderForTiesAndDeduplicatesIds() {
        List<ReciprocalRankFusion.FusedResult<String>> fused = ReciprocalRankFusion.fuse(
                List.of(List.of("a", "a", "b"), List.of("c")),
                value -> value,
                10,
                3);

        assertThat(fused).extracting(ReciprocalRankFusion.FusedResult::value)
                .containsExactly("a", "c", "b");
    }

    @Test
    void supportsPerRetrieverWeights() {
        List<ReciprocalRankFusion.FusedResult<String>> fused = ReciprocalRankFusion.fuse(
                List.of(List.of("dense", "shared"), List.of("bm25", "shared")),
                List.of(0.5d, 1.0d),
                value -> value,
                60,
                3);

        assertThat(fused).extracting(ReciprocalRankFusion.FusedResult::value)
                .containsExactly("shared", "bm25", "dense");
    }

    @Test
    void rejectsInvalidArguments() {
        assertThatThrownBy(() -> ReciprocalRankFusion.<String>fuse(List.of(), value -> value, -1, 10))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> ReciprocalRankFusion.<String>fuse(List.of(), value -> value, 60, 0))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> ReciprocalRankFusion.fuse(
                List.of(List.of("a")), List.of(), value -> value, 60, 10))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
