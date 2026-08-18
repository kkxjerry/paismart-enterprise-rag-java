package com.yizhaoqi.smartpai.benchmark;

import java.util.ArrayList;
import java.util.List;

final class TextChunker {

    private static final List<String> SEPARATORS = List.of(
            "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ");

    private TextChunker() {
    }

    static List<String> chunk(String text, int chunkSize, int overlap) {
        if (chunkSize <= 0 || overlap < 0 || overlap >= chunkSize) {
            throw new IllegalArgumentException("chunk size must be positive and overlap must be smaller");
        }
        String cleaned = clean(text);
        if (cleaned.isBlank()) {
            return List.of();
        }

        List<String> chunks = new ArrayList<>();
        int start = 0;
        while (start < cleaned.length()) {
            int targetEnd = Math.min(start + chunkSize, cleaned.length());
            int end = targetEnd;
            if (targetEnd < cleaned.length()) {
                String window = cleaned.substring(start, targetEnd);
                int best = -1;
                for (String separator : SEPARATORS) {
                    best = Math.max(best, window.lastIndexOf(separator));
                }
                if (best >= (int) (chunkSize * 0.6d)) {
                    end = start + best + 1;
                }
            }
            String chunk = cleaned.substring(start, end).trim();
            if (!chunk.isBlank()) {
                chunks.add(chunk);
            }
            if (end >= cleaned.length()) {
                break;
            }
            start = end - overlap;
        }
        return List.copyOf(chunks);
    }

    private static String clean(String text) {
        if (text == null) {
            return "";
        }
        return text.replace("\r\n", "\n")
                .replace('\r', '\n')
                .replaceAll("[ \\t]+", " ")
                .replaceAll("\\n{3,}", "\n\n")
                .trim();
    }
}
