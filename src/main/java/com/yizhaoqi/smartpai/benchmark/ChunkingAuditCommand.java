package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.BufferedReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Runs source chunking without embeddings or index writes. */
public final class ChunkingAuditCommand {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private ChunkingAuditCommand() {
    }

    public static void main(String[] args) throws Exception {
        Arguments values = Arguments.parse(args);
        Path docs = Path.of(values.required("docs"));
        String strategy = values.string("chunking-strategy", "fixed").toLowerCase(Locale.ROOT);
        if (!Set.of("fixed", "source-aware", "parent-child").contains(strategy)) {
            throw new IllegalArgumentException("invalid chunking strategy: " + strategy);
        }
        int chunkSize = values.positiveInt("chunk-size", 1200);
        int overlap = values.nonNegativeInt("chunk-overlap", 200);
        if (overlap >= chunkSize) {
            throw new IllegalArgumentException("chunk overlap must be smaller than chunk size");
        }
        long maximum = Long.parseLong(values.string("max-documents", "0"));
        if (maximum < 0) {
            throw new IllegalArgumentException("max documents must not be negative");
        }
        Set<String> selectedSources = new HashSet<>();
        for (String raw : values.string("source-aware-types", "").split(",")) {
            if (!raw.isBlank()) {
                selectedSources.add(raw.trim().toLowerCase(Locale.ROOT));
            }
        }
        Map<String, SourceStats> bySource = new LinkedHashMap<>();
        List<Integer> chunksPerDocument = new ArrayList<>();
        long documents = 0;
        long chunks = 0;
        long parents = 0;
        long rawCharacters = 0;
        long leafCharacters = 0;
        long uniqueParentCharacters = 0;
        try (BufferedReader reader = Files.newBufferedReader(docs, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                JsonNode row = MAPPER.readTree(line);
                String source = row.path("source_type").asText("unknown").toLowerCase(Locale.ROOT);
                String title = row.path("title").asText("");
                String text = row.path("text").asText("");
                List<SourceAwareChunker.Segment> segments = segments(
                        strategy, selectedSources, source, title, text, chunkSize, overlap);
                Set<String> parentIds = new HashSet<>();
                long documentParentCharacters = 0;
                long documentLeafCharacters = 0;
                for (int index = 0; index < segments.size(); index++) {
                    SourceAwareChunker.Segment segment = segments.get(index);
                    if (!segment.parentText().isBlank()) {
                        if (segment.parentStart() < 0 || segment.parentEnd() < segment.parentStart()
                                || segment.parentEnd() > segment.parentText().length()
                                || !segment.text().equals(segment.parentText().substring(
                                        segment.parentStart(), segment.parentEnd()))) {
                            throw new IllegalStateException(
                                    "leaf is not an exact parent substring for document "
                                            + row.path("doc_id").asText());
                        }
                    }
                    String parentId = segment.parentId().isBlank()
                            ? "leaf-" + index
                            : segment.parentId();
                    if (parentIds.add(parentId)) {
                        documentParentCharacters += segment.parentText().isBlank()
                                ? segment.text().length()
                                : segment.parentText().length();
                    }
                    leafCharacters += segment.text().length();
                    documentLeafCharacters += segment.text().length();
                }
                documents++;
                chunks += segments.size();
                parents += parentIds.size();
                rawCharacters += text.length();
                uniqueParentCharacters += documentParentCharacters;
                chunksPerDocument.add(segments.size());
                bySource.computeIfAbsent(source, ignored -> new SourceStats())
                        .observe(text.length(), segments.size(), parentIds.size(),
                                documentParentCharacters, documentLeafCharacters);
                if (maximum > 0 && documents >= maximum) {
                    break;
                }
            }
        }
        ObjectNode output = MAPPER.createObjectNode();
        output.put("documents", documents);
        output.put("chunks", chunks);
        output.put("parents", parents);
        output.put("chunks_per_document_mean", ratio(chunks, documents));
        output.put("chunks_per_document_p50", percentile(chunksPerDocument, 0.50));
        output.put("chunks_per_document_p95", percentile(chunksPerDocument, 0.95));
        output.put("parents_per_document_mean", ratio(parents, documents));
        output.put("raw_characters", rawCharacters);
        output.put("leaf_characters", leafCharacters);
        output.put("unique_parent_characters", uniqueParentCharacters);
        output.put("leaf_to_raw_character_ratio", ratio(leafCharacters, rawCharacters));
        output.put("unique_parent_to_raw_character_ratio", ratio(uniqueParentCharacters, rawCharacters));
        output.put("strategy", strategy);
        output.put("chunk_size", chunkSize);
        output.put("chunk_overlap", overlap);
        ArrayNode selected = output.putArray("selected_source_types");
        selectedSources.stream().sorted().forEach(selected::add);
        ObjectNode sourceNode = output.putObject("by_source_type");
        bySource.entrySet().stream().sorted(Map.Entry.comparingByKey()).forEach(entry ->
                sourceNode.set(entry.getKey(), entry.getValue().toJson()));
        String rendered = MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(output);
        System.out.println(rendered);
        String outputPath = values.string("output", "");
        if (!outputPath.isBlank()) {
            Path path = Path.of(outputPath);
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            Files.writeString(path, rendered + System.lineSeparator(), StandardCharsets.UTF_8);
        }
    }

    private static List<SourceAwareChunker.Segment> segments(
            String strategy,
            Set<String> selectedSources,
            String source,
            String title,
            String text,
            int chunkSize,
            int overlap) {
        boolean selected = selectedSources.isEmpty() || selectedSources.contains(source);
        if ("parent-child".equals(strategy)
                && selected
                && HierarchicalSourceChunker.supportedTypes().contains(source)) {
            return HierarchicalSourceChunker.chunk(source, title, text, chunkSize, overlap);
        }
        if ("source-aware".equals(strategy) && selected) {
            return SourceAwareChunker.chunk(source, title, text, chunkSize, overlap);
        }
        return TextChunker.chunk(text, chunkSize, overlap).stream()
                .map(value -> new SourceAwareChunker.Segment(value, "body", title, "", "", ""))
                .toList();
    }

    private static double ratio(long numerator, long denominator) {
        return denominator == 0 ? 0.0d : numerator / (double) denominator;
    }

    private static int percentile(List<Integer> values, double percentile) {
        if (values.isEmpty()) {
            return 0;
        }
        List<Integer> sorted = values.stream().sorted(Comparator.naturalOrder()).toList();
        int index = (int) Math.ceil(percentile * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(index, sorted.size() - 1)));
    }

    private static final class SourceStats {
        private long documents;
        private long rawCharacters;
        private long chunks;
        private long parents;
        private long parentCharacters;
        private long leafCharacters;

        private void observe(
                long raw,
                long chunkCount,
                long parentCount,
                long parentChars,
                long leafChars) {
            documents++;
            rawCharacters += raw;
            chunks += chunkCount;
            parents += parentCount;
            parentCharacters += parentChars;
            leafCharacters += leafChars;
        }

        private ObjectNode toJson() {
            ObjectNode value = MAPPER.createObjectNode();
            value.put("documents", documents);
            value.put("chunks", chunks);
            value.put("parents", parents);
            value.put("chunks_per_document", ratio(chunks, documents));
            value.put("parents_per_document", ratio(parents, documents));
            value.put("leaf_to_raw_character_ratio", ratio(leafCharacters, rawCharacters));
            value.put("unique_parent_to_raw_character_ratio", ratio(parentCharacters, rawCharacters));
            return value;
        }
    }
}
