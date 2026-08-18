package com.yizhaoqi.smartpai.service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Combines independent retrieval rankings without comparing incompatible raw scores.
 */
public final class ReciprocalRankFusion {

    private ReciprocalRankFusion() {
    }

    public static <T> List<FusedResult<T>> fuse(
            List<? extends List<T>> rankings,
            Function<T, String> idExtractor,
            int rrfK,
            int limit) {
        return fuse(rankings, rankings.stream().map(ignored -> 1.0d).toList(), idExtractor, rrfK, limit);
    }

    public static <T> List<FusedResult<T>> fuse(
            List<? extends List<T>> rankings,
            List<Double> weights,
            Function<T, String> idExtractor,
            int rrfK,
            int limit) {
        if (rrfK < 0) {
            throw new IllegalArgumentException("rrfK must not be negative");
        }
        if (limit <= 0) {
            throw new IllegalArgumentException("limit must be positive");
        }
        if (rankings.size() != weights.size()) {
            throw new IllegalArgumentException("rankings and weights must have the same size");
        }
        if (weights.stream().anyMatch(weight -> weight == null || weight <= 0.0d)) {
            throw new IllegalArgumentException("weights must be positive");
        }

        Map<String, Double> scores = new HashMap<>();
        Map<String, Integer> firstSeen = new HashMap<>();
        Map<String, T> values = new LinkedHashMap<>();
        int seenOrder = 0;

        for (int rankingIndex = 0; rankingIndex < rankings.size(); rankingIndex++) {
            List<T> ranking = rankings.get(rankingIndex);
            double weight = weights.get(rankingIndex);
            for (int index = 0; index < ranking.size(); index++) {
                T value = ranking.get(index);
                String id = idExtractor.apply(value);
                if (id == null || id.isBlank()) {
                    continue;
                }
                if (!firstSeen.containsKey(id)) {
                    firstSeen.put(id, seenOrder++);
                    values.put(id, value);
                }
                int rank = index + 1;
                scores.merge(id, weight / (rrfK + rank), Double::sum);
            }
        }

        return scores.entrySet().stream()
                .sorted(Comparator
                        .<Map.Entry<String, Double>>comparingDouble(Map.Entry::getValue)
                        .reversed()
                        .thenComparingInt(entry -> firstSeen.get(entry.getKey())))
                .limit(limit)
                .map(entry -> new FusedResult<>(values.get(entry.getKey()), entry.getValue()))
                .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
    }

    public record FusedResult<T>(T value, double score) {
    }
}
