package com.yizhaoqi.smartpai.benchmark;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Source-shaped chunking that preserves the structural unit used to produce a
 * chunk. It deliberately falls back to {@link TextChunker} when a source does
 * not expose a reliable boundary.
 */
final class SourceAwareChunker {

    private static final Pattern MARKDOWN_HEADING = Pattern.compile("(?m)^(#{1,6})\\s+(.+?)\\s*$");
    private static final Pattern JIRA_SECTION = Pattern.compile(
            "(?im)^(summary|issue summary|description|impact|steps to reproduce|root cause|cause|"
                    + "workaround|temporary workaround|resolution|acceptance criteria|comments?|logs?[^:]*)\\s*:?\\s*$");
    private static final Pattern SPEAKER_TURN = Pattern.compile(
            "^\\s*(?:\\[(?<time>\\d{1,2}:\\d{2}(?::\\d{2})?)\\]\\s*)?"
                    + "(?<speaker>[^:]{1,80}):\\s*(?<body>.*)$");
    private static final Pattern SLACK_PREFIX = Pattern.compile(
            "^\\s*(?:\\[(?<time>[^]]+)\\]|(?<time2>\\d{4}-\\d{2}-\\d{2}[^ ]*))?\\s*"
                    + "(?<speaker>[^:]{1,80}):\\s*(?<body>.*)$");

    private static final Set<String> SUPPORTED_TYPES = Set.of(
            "jira", "linear", "slack", "fireflies", "confluence", "google_drive", "github");

    private SourceAwareChunker() {
    }

    static Set<String> supportedTypes() {
        return SUPPORTED_TYPES;
    }

    static List<Segment> chunk(
            String sourceType,
            String title,
            String text,
            int chunkSize,
            int overlap) {
        String type = sourceType == null ? "" : sourceType.toLowerCase(Locale.ROOT);
        String cleaned = normalize(text);
        if (cleaned.isBlank()) {
            return List.of();
        }
        List<Block> blocks = switch (type) {
            case "jira", "linear" -> jiraBlocks(cleaned);
            case "slack" -> conversationBlocks(cleaned, SLACK_PREFIX, "thread");
            case "fireflies" -> conversationBlocks(cleaned, SPEAKER_TURN, "transcript");
            case "confluence", "google_drive" -> markdownBlocks(cleaned, "section");
            case "github" -> githubBlocks(cleaned);
            default -> List.of(new Block(cleaned, "body", title == null ? "" : title, "", "", ""));
        };
        blocks = coalesceBlocks(blocks, chunkSize);
        List<Segment> segments = new ArrayList<>();
        for (Block block : blocks) {
            if (block.text().isBlank()) {
                continue;
            }
            List<String> parts = TextChunker.chunk(block.text(), chunkSize, overlap);
            for (int index = 0; index < parts.size(); index++) {
                String section = block.sectionPath();
                if (parts.size() > 1) {
                    section = section.isBlank()
                            ? "part-" + (index + 1)
                            : section + " / part-" + (index + 1);
                }
                segments.add(new Segment(
                        parts.get(index),
                        block.kind(),
                        section,
                        block.speaker(),
                        block.threadId(),
                        block.eventTime()));
            }
        }
        if (segments.isEmpty()) {
            return TextChunker.chunk(cleaned, chunkSize, overlap).stream()
                    .map(value -> new Segment(value, "body", title == null ? "" : title, "", "", ""))
                    .toList();
        }
        return List.copyOf(segments);
    }

    private static List<Block> jiraBlocks(String text) {
        Matcher matcher = JIRA_SECTION.matcher(text);
        List<Boundary> boundaries = new ArrayList<>();
        while (matcher.find()) {
            boundaries.add(new Boundary(matcher.start(), matcher.end(), normalizeLabel(matcher.group(1))));
        }
        if (boundaries.isEmpty()) {
            return markdownBlocks(text, "issue");
        }
        List<Block> blocks = new ArrayList<>();
        if (boundaries.get(0).start() > 0) {
            String prefix = text.substring(0, boundaries.get(0).start()).trim();
            if (!prefix.isBlank()) {
                blocks.add(new Block(prefix, "issue_header", "header", "", "", ""));
            }
        }
        for (int index = 0; index < boundaries.size(); index++) {
            Boundary boundary = boundaries.get(index);
            int end = index + 1 < boundaries.size() ? boundaries.get(index + 1).start() : text.length();
            String body = text.substring(boundary.end(), end).trim();
            if (!body.isBlank()) {
                blocks.add(new Block(body, "issue_" + slug(boundary.label()), boundary.label(), "", "", ""));
            }
        }
        return blocks;
    }

    private static List<Block> markdownBlocks(String text, String defaultKind) {
        Matcher matcher = MARKDOWN_HEADING.matcher(text);
        List<Boundary> headings = new ArrayList<>();
        while (matcher.find()) {
            headings.add(new Boundary(matcher.start(), matcher.end(), matcher.group(2).trim()));
        }
        if (headings.isEmpty()) {
            return paragraphBlocks(text, defaultKind, "");
        }
        List<Block> blocks = new ArrayList<>();
        if (headings.get(0).start() > 0) {
            blocks.addAll(paragraphBlocks(text.substring(0, headings.get(0).start()), defaultKind, "preamble"));
        }
        for (int index = 0; index < headings.size(); index++) {
            Boundary heading = headings.get(index);
            int end = index + 1 < headings.size() ? headings.get(index + 1).start() : text.length();
            String body = text.substring(heading.end(), end).trim();
            if (!body.isBlank()) {
                blocks.add(new Block(body, defaultKind, heading.label(), "", "", ""));
            }
        }
        return blocks;
    }

    private static List<Block> githubBlocks(String text) {
        List<Block> markdown = markdownBlocks(text, "github_section");
        List<Block> output = new ArrayList<>();
        for (Block block : markdown) {
            if (!block.text().contains("```")) {
                output.add(block);
                continue;
            }
            String[] pieces = block.text().split("(?m)(?=^```)|(?m)(?<=^```\\s*$)");
            for (String piece : pieces) {
                String trimmed = piece.trim();
                if (trimmed.isBlank()) {
                    continue;
                }
                output.add(new Block(
                        trimmed,
                        trimmed.startsWith("```") ? "code_block" : block.kind(),
                        block.sectionPath(),
                        "",
                        "",
                        ""));
            }
        }
        return output;
    }

    private static List<Block> conversationBlocks(String text, Pattern pattern, String threadId) {
        List<Block> blocks = new ArrayList<>();
        String currentSpeaker = "";
        String currentTime = "";
        StringBuilder current = new StringBuilder();
        int turn = 0;
        for (String line : text.split("\\n")) {
            Matcher matcher = pattern.matcher(line);
            if (matcher.matches()) {
                flushConversation(blocks, current, currentSpeaker, currentTime, threadId, turn);
                current.setLength(0);
                currentSpeaker = safeGroup(matcher, "speaker");
                currentTime = safeGroup(matcher, "time");
                if (currentTime.isBlank()) {
                    currentTime = safeGroup(matcher, "time2");
                }
                String body = safeGroup(matcher, "body");
                current.append(body);
                turn++;
            } else if (!line.isBlank()) {
                if (!current.isEmpty()) {
                    current.append('\n');
                }
                current.append(line.trim());
            }
        }
        flushConversation(blocks, current, currentSpeaker, currentTime, threadId, turn);
        return blocks.isEmpty() ? paragraphBlocks(text, "conversation", threadId) : blocks;
    }

    private static void flushConversation(
            List<Block> blocks,
            StringBuilder text,
            String speaker,
            String eventTime,
            String threadId,
            int turn) {
        String value = text.toString().trim();
        if (!value.isBlank()) {
            blocks.add(new Block(
                    value,
                    "conversation_turn",
                    "turn-" + Math.max(1, turn),
                    speaker,
                    threadId,
                    eventTime));
        }
    }

    private static List<Block> coalesceBlocks(List<Block> blocks, int chunkSize) {
        if (blocks.size() < 2) {
            return blocks;
        }
        List<Block> output = new ArrayList<>();
        List<Block> pending = new ArrayList<>();
        int pendingLength = 0;
        for (Block block : blocks) {
            if (!mergeable(block)) {
                flushCoalesced(output, pending);
                pendingLength = 0;
                output.add(block);
                continue;
            }
            String rendered = renderForMerge(block);
            int separator = pending.isEmpty() ? 0 : 2;
            if (!pending.isEmpty() && pendingLength + separator + rendered.length() > chunkSize) {
                flushCoalesced(output, pending);
                pendingLength = 0;
            }
            pending.add(block);
            pendingLength += (pending.size() == 1 ? 0 : separator) + rendered.length();
        }
        flushCoalesced(output, pending);
        return output;
    }

    private static boolean mergeable(Block block) {
        return "conversation_turn".equals(block.kind())
                || "conversation".equals(block.kind())
                || block.sectionPath().contains("paragraph-");
    }

    private static void flushCoalesced(List<Block> output, List<Block> pending) {
        if (pending.isEmpty()) {
            return;
        }
        if (pending.size() == 1) {
            output.add(pending.get(0));
            pending.clear();
            return;
        }
        Block first = pending.get(0);
        Block last = pending.get(pending.size() - 1);
        String text = pending.stream().map(SourceAwareChunker::renderForMerge)
                .reduce((left, right) -> left + "\n\n" + right)
                .orElse("");
        String speakers = pending.stream()
                .map(Block::speaker)
                .filter(value -> !value.isBlank())
                .distinct()
                .reduce((left, right) -> left + "," + right)
                .orElse("");
        String section = first.sectionPath().equals(last.sectionPath())
                ? first.sectionPath()
                : first.sectionPath() + " … " + last.sectionPath();
        output.add(new Block(
                text,
                first.kind().equals("conversation_turn") ? "conversation_window" : first.kind(),
                section,
                speakers,
                first.threadId(),
                first.eventTime()));
        pending.clear();
    }

    private static String renderForMerge(Block block) {
        if (!"conversation_turn".equals(block.kind())) {
            return block.text();
        }
        StringBuilder prefix = new StringBuilder();
        if (!block.eventTime().isBlank()) {
            prefix.append('[').append(block.eventTime()).append("] ");
        }
        if (!block.speaker().isBlank()) {
            prefix.append(block.speaker()).append(": ");
        }
        return prefix + block.text();
    }

    private static List<Block> paragraphBlocks(String text, String kind, String section) {
        List<Block> blocks = new ArrayList<>();
        int index = 0;
        for (String paragraph : text.split("\\n\\s*\\n")) {
            String value = paragraph.trim();
            if (!value.isBlank()) {
                index++;
                blocks.add(new Block(
                        value,
                        kind,
                        section.isBlank() ? "paragraph-" + index : section + " / paragraph-" + index,
                        "",
                        "",
                        ""));
            }
        }
        return blocks;
    }

    private static String safeGroup(Matcher matcher, String name) {
        try {
            String value = matcher.group(name);
            return value == null ? "" : value.trim();
        } catch (IllegalArgumentException ignored) {
            return "";
        }
    }

    private static String normalizeLabel(String value) {
        String normalized = value == null ? "section" : value.trim().replaceAll("\\s+", " ");
        return normalized.isBlank() ? "section" : normalized;
    }

    private static String slug(String value) {
        return normalizeLabel(value).toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "_")
                .replaceAll("^_+|_+$", "");
    }

    private static String normalize(String text) {
        if (text == null) {
            return "";
        }
        return text.replace("\r\n", "\n").replace('\r', '\n').replaceAll("[ \\t]+", " ").trim();
    }

    record Segment(
            String text,
            String kind,
            String sectionPath,
            String speaker,
            String threadId,
            String eventTime) {
    }

    private record Block(
            String text,
            String kind,
            String sectionPath,
            String speaker,
            String threadId,
            String eventTime) {
    }

    private record Boundary(int start, int end, String label) {
    }
}
