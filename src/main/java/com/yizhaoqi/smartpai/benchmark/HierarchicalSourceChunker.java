package com.yizhaoqi.smartpai.benchmark;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Experimental E3 source-specific leaf/parent chunker.
 *
 * <p>The indexed leaf remains a verbatim substring of {@code parentText}. The
 * deterministic context prefix is for retrieval only. Generation and citation
 * can recover the parent without synthesising text.</p>
 */
final class HierarchicalSourceChunker {

    private static final Set<String> SUPPORTED_TYPES = Set.of(
            "confluence", "google_drive", "gmail", "fireflies");
    private static final Pattern MARKDOWN_HEADING = Pattern.compile("(?m)^(#{1,6})\\s+(.+?)\\s*$");
    private static final Pattern FIREFLIES_SECTION = Pattern.compile(
            "(?im)^(summary|transcript|topics?|next[_ ]steps?|action[_ ]items?|decisions?|"
                    + "meeting header|attendees|timeline)\\s*:\\s*$");
    private static final Pattern GMAIL_MESSAGE = Pattern.compile("(?m)^From:\\s+.+$");
    private static final Pattern GMAIL_HEADER = Pattern.compile(
            "(?im)^(From|To|Cc|Date|Subject):\\s*(.+?)\\s*$");
    private static final Pattern TRANSCRIPT_TURN = Pattern.compile(
            "(?m)^\\s*\\[(?<time>[^]]+)]\\s*(?<speaker>[^:]{1,100}):\\s*(?<body>.*)$");
    private static final Pattern LIST_OR_TABLE = Pattern.compile(
            "^\\s*(?:[-*+]\\s+|\\d+[.)]\\s+|\\|.*\\|\\s*$)");
    private static final List<String> CUT_SEPARATORS = List.of(
            "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ");

    private HierarchicalSourceChunker() {
    }

    static Set<String> supportedTypes() {
        return SUPPORTED_TYPES;
    }

    static List<SourceAwareChunker.Segment> chunk(
            String sourceType,
            String title,
            String text,
            int leafSize,
            int overlap) {
        if (leafSize <= 0 || overlap < 0 || overlap >= leafSize) {
            throw new IllegalArgumentException("leaf size must be positive and overlap must be smaller");
        }
        String source = value(sourceType).toLowerCase(Locale.ROOT);
        String normalized = normalize(text);
        if (normalized.isBlank()) {
            return List.of();
        }
        int parentSize = Math.max(1_600, Math.min(2_400, leafSize * 4));
        List<Parent> parents = switch (source) {
            case "confluence", "google_drive" -> markdownWindowParents(normalized, parentSize);
            case "gmail" -> gmailParents(normalized);
            case "fireflies" -> firefliesParents(normalized);
            default -> List.of(new Parent("parent-1", "body", value(title), "", "", normalized));
        };
        parents = boundParents(parents, parentSize);
        List<SourceAwareChunker.Segment> output = new ArrayList<>();
        int parentIndex = 0;
        for (Parent parent : parents) {
            parentIndex++;
            List<Leaf> leaves = leaves(parent, leafSize, overlap);
            int leafIndex = 0;
            for (Leaf leaf : leaves) {
                leafIndex++;
                String prefix = contextPrefix(
                        title,
                        source,
                        parent.sectionPath(),
                        parent.kind(),
                        leaf.speaker(),
                        leaf.eventTime());
                output.add(new SourceAwareChunker.Segment(
                        leaf.text(),
                        leaf.kind(),
                        parent.sectionPath(),
                        leaf.speaker(),
                        parent.threadId(),
                        leaf.eventTime(),
                        "parent-" + parentIndex,
                        parent.text(),
                        prefix,
                        leaf.start(),
                        leaf.end()));
            }
        }
        if (output.isEmpty()) {
            String prefix = contextPrefix(title, source, value(title), "body", "", "");
            return List.of(new SourceAwareChunker.Segment(
                    normalized,
                    "body",
                    value(title),
                    "",
                    "",
                    "",
                    "parent-1",
                    normalized,
                    prefix,
                    0,
                    normalized.length()));
        }
        return List.copyOf(output);
    }

    private static List<Parent> markdownWindowParents(String text, int parentSize) {
        Matcher matcher = MARKDOWN_HEADING.matcher(text);
        List<Heading> headings = new ArrayList<>();
        while (matcher.find()) {
            headings.add(new Heading(
                    matcher.start(),
                    matcher.end(),
                    matcher.group(1).length(),
                    matcher.group(2).trim()));
        }
        List<Range> windows = exactSlices(text, 0, text.length(), parentSize, 0);
        List<Parent> output = new ArrayList<>();
        for (int index = 0; index < windows.size(); index++) {
            Range window = windows.get(index);
            String raw = text.substring(window.start(), window.end());
            int left = 0;
            int right = raw.length();
            while (left < right && Character.isWhitespace(raw.charAt(left))) {
                left++;
            }
            while (right > left && Character.isWhitespace(raw.charAt(right - 1))) {
                right--;
            }
            if (left >= right) {
                continue;
            }
            int absoluteStart = window.start() + left;
            int absoluteEnd = window.start() + right;
            String section = markdownSectionAt(headings, absoluteStart, absoluteEnd);
            output.add(new Parent(
                    "markdown-window-" + (index + 1),
                    "section_window",
                    section,
                    "",
                    "",
                    text.substring(absoluteStart, absoluteEnd)));
        }
        return output;
    }

    private static String markdownSectionAt(List<Heading> headings, int start, int end) {
        List<String> stack = new ArrayList<>();
        List<String> observed = new ArrayList<>();
        for (Heading heading : headings) {
            if (heading.start() >= end) {
                break;
            }
            while (stack.size() >= heading.level()) {
                stack.remove(stack.size() - 1);
            }
            stack.add(heading.label());
            if (heading.end() > start) {
                String path = String.join(" / ", stack);
                if (!observed.contains(path)) {
                    observed.add(path);
                }
            }
        }
        if (!observed.isEmpty()) {
            return observed.size() == 1
                    ? observed.get(0)
                    : observed.get(0) + " … " + observed.get(observed.size() - 1);
        }
        return stack.isEmpty() ? "body" : String.join(" / ", stack);
    }

    private static List<Parent> firefliesParents(String text) {
        Matcher matcher = FIREFLIES_SECTION.matcher(text);
        List<Boundary> sections = new ArrayList<>();
        while (matcher.find()) {
            sections.add(new Boundary(matcher.start(), matcher.end(), normalizeLabel(matcher.group(1))));
        }
        if (sections.isEmpty()) {
            return paragraphParents(text, "meeting_section", "meeting");
        }
        List<Parent> output = new ArrayList<>();
        if (sections.get(0).start() > 0) {
            String preamble = text.substring(0, sections.get(0).start()).trim();
            if (!preamble.isBlank()) {
                output.add(new Parent("meeting-preamble", "meeting_section", "preamble", "", "meeting", preamble));
            }
        }
        for (int index = 0; index < sections.size(); index++) {
            Boundary section = sections.get(index);
            int end = index + 1 < sections.size() ? sections.get(index + 1).start() : text.length();
            String body = text.substring(section.end(), end).trim();
            if (!body.isBlank()) {
                output.add(new Parent(
                        "meeting-" + (index + 1),
                        "meeting_" + slug(section.label()),
                        section.label(),
                        "",
                        "meeting",
                        body));
            }
        }
        return output;
    }

    private static List<Parent> gmailParents(String text) {
        Matcher matcher = GMAIL_MESSAGE.matcher(text);
        List<Integer> starts = new ArrayList<>();
        while (matcher.find()) {
            starts.add(matcher.start());
        }
        if (starts.isEmpty()) {
            return paragraphParents(text, "email_message", "message");
        }
        List<Parent> output = new ArrayList<>();
        for (int index = 0; index < starts.size(); index++) {
            int start = starts.get(index);
            int end = index + 1 < starts.size() ? starts.get(index + 1) : text.length();
            String message = text.substring(start, end).trim();
            if (message.isBlank()) {
                continue;
            }
            String subject = header(message, "subject");
            String sender = header(message, "from");
            String date = header(message, "date");
            String section = subject.isBlank() ? "email message " + (index + 1) : subject;
            output.add(new Parent(
                    "email-" + (index + 1),
                    "email_message",
                    section,
                    sender,
                    "email-thread",
                    message,
                    date));
        }
        return output;
    }

    private static List<Parent> paragraphParents(String text, String kind, String prefix) {
        List<Parent> output = new ArrayList<>();
        int index = 0;
        for (Range range : paragraphRanges(text)) {
            String paragraph = text.substring(range.start(), range.end()).trim();
            if (!paragraph.isBlank()) {
                index++;
                output.add(new Parent(
                        prefix + "-" + index,
                        kind,
                        prefix + " / paragraph-" + index,
                        "",
                        "",
                        paragraph));
            }
        }
        return output;
    }

    private static List<Parent> boundParents(List<Parent> parents, int parentSize) {
        List<Parent> output = new ArrayList<>();
        for (Parent parent : parents) {
            if (parent.text().length() <= parentSize) {
                output.add(parent);
                continue;
            }
            List<Range> pieces = exactSlices(parent.text(), 0, parent.text().length(), parentSize, 0);
            int index = 0;
            for (Range piece : pieces) {
                String text = parent.text().substring(piece.start(), piece.end()).trim();
                if (text.isBlank()) {
                    continue;
                }
                index++;
                String section = parent.sectionPath() + " / parent-part-" + index;
                output.add(new Parent(
                        parent.id() + "-part-" + index,
                        parent.kind(),
                        section,
                        parent.speaker(),
                        parent.threadId(),
                        text,
                        parent.eventTime()));
            }
        }
        return output;
    }

    private static List<Leaf> leaves(Parent parent, int leafSize, int overlap) {
        if (parent.kind().equals("meeting_transcript")) {
            List<Leaf> turns = transcriptLeaves(parent, leafSize);
            if (!turns.isEmpty()) {
                return turns;
            }
        }
        List<Leaf> output = new ArrayList<>();
        String kind = isStructured(parent.text()) ? "structured_leaf" : "leaf_text";
        for (Range piece : exactSlices(parent.text(), 0, parent.text().length(), leafSize, overlap)) {
            addLeaf(output, parent, piece, kind, parent.speaker(), parent.eventTime());
        }
        return output;
    }

    private static List<Leaf> transcriptLeaves(Parent parent, int leafSize) {
        Matcher matcher = TRANSCRIPT_TURN.matcher(parent.text());
        List<TranscriptRange> ranges = new ArrayList<>();
        while (matcher.find()) {
            ranges.add(new TranscriptRange(
                    matcher.start(),
                    matcher.end(),
                    value(matcher.group("speaker")),
                    value(matcher.group("time"))));
        }
        List<Leaf> output = new ArrayList<>();
        if (ranges.isEmpty()) {
            return output;
        }
        int start = ranges.get(0).start();
        int end = ranges.get(0).end();
        List<String> speakers = new ArrayList<>();
        if (!ranges.get(0).speaker().isBlank()) {
            speakers.add(ranges.get(0).speaker());
        }
        String firstTime = ranges.get(0).time();
        for (int index = 1; index < ranges.size(); index++) {
            TranscriptRange next = ranges.get(index);
            if (next.end() - start > leafSize && end > start) {
                addLeaf(output, parent, new Range(start, end), "transcript_window",
                        String.join(",", speakers), firstTime);
                start = next.start();
                speakers.clear();
                firstTime = next.time();
            }
            end = next.end();
            if (!next.speaker().isBlank() && !speakers.contains(next.speaker())) {
                speakers.add(next.speaker());
            }
        }
        addLeaf(output, parent, new Range(start, end), "transcript_window",
                String.join(",", speakers), firstTime);
        return output;
    }

    private static List<Leaf> structuredLeaves(Parent parent, Range paragraph, int leafSize) {
        List<Range> ranges = new ArrayList<>();
        int cursor = paragraph.start();
        String block = parent.text().substring(paragraph.start(), paragraph.end());
        for (String raw : block.split("\\n", -1)) {
            int end = cursor + raw.length();
            if (!raw.isBlank()) {
                ranges.add(new Range(cursor, end));
            }
            cursor = Math.min(parent.text().length(), end + 1);
        }
        String kind = block.stripLeading().startsWith("|") ? "table_rows" : "list_items";
        return groupedLeaves(parent, ranges, leafSize, kind);
    }

    private static List<Leaf> groupedLeaves(
            Parent parent,
            List<Range> ranges,
            int leafSize,
            String kind) {
        List<Leaf> output = new ArrayList<>();
        if (ranges.isEmpty()) {
            return output;
        }
        int start = ranges.get(0).start();
        int end = ranges.get(0).end();
        for (int index = 1; index < ranges.size(); index++) {
            Range next = ranges.get(index);
            if (next.end() - start > leafSize && end > start) {
                addLeaf(output, parent, new Range(start, end), kind, parent.speaker(), parent.eventTime());
                start = next.start();
            }
            end = next.end();
        }
        addLeaf(output, parent, new Range(start, end), kind, parent.speaker(), parent.eventTime());
        return output;
    }

    private static void addLeaf(
            List<Leaf> output,
            Parent parent,
            Range range,
            String kind,
            String speaker,
            String eventTime) {
        int start = range.start();
        int end = range.end();
        while (start < end && Character.isWhitespace(parent.text().charAt(start))) {
            start++;
        }
        while (end > start && Character.isWhitespace(parent.text().charAt(end - 1))) {
            end--;
        }
        if (start < end) {
            output.add(new Leaf(
                    parent.text().substring(start, end),
                    kind,
                    start,
                    end,
                    value(speaker),
                    value(eventTime)));
        }
    }

    private static List<Range> exactSlices(
            String text,
            int rangeStart,
            int rangeEnd,
            int leafSize,
            int overlap) {
        List<Range> output = new ArrayList<>();
        int start = rangeStart;
        while (start < rangeEnd) {
            int target = Math.min(start + leafSize, rangeEnd);
            int end = target;
            if (target < rangeEnd) {
                String window = text.substring(start, target);
                int best = -1;
                int separatorLength = 0;
                for (String separator : CUT_SEPARATORS) {
                    int index = window.lastIndexOf(separator);
                    if (index > best) {
                        best = index;
                        separatorLength = separator.length();
                    }
                }
                if (best >= (int) (leafSize * 0.55d)) {
                    end = start + best + separatorLength;
                }
            }
            output.add(new Range(start, end));
            if (end >= rangeEnd) {
                break;
            }
            int next = Math.max(start + 1, end - overlap);
            start = next;
        }
        return output;
    }

    private static List<Range> paragraphRanges(String text) {
        List<Range> output = new ArrayList<>();
        Matcher matcher = Pattern.compile("\\n[ \\t]*\\n").matcher(text);
        int start = 0;
        while (matcher.find()) {
            if (!text.substring(start, matcher.start()).isBlank()) {
                output.add(new Range(start, matcher.start()));
            }
            start = matcher.end();
        }
        if (start < text.length() && !text.substring(start).isBlank()) {
            output.add(new Range(start, text.length()));
        }
        return output.isEmpty() && !text.isBlank() ? List.of(new Range(0, text.length())) : output;
    }

    private static boolean isStructured(String value) {
        int structured = 0;
        int nonBlank = 0;
        for (String line : value.split("\\n")) {
            if (line.isBlank()) {
                continue;
            }
            nonBlank++;
            if (LIST_OR_TABLE.matcher(line).find()) {
                structured++;
            }
        }
        return nonBlank > 0 && structured >= Math.max(1, nonBlank / 2);
    }

    private static String header(String message, String name) {
        Matcher matcher = GMAIL_HEADER.matcher(message);
        while (matcher.find()) {
            if (matcher.group(1).equalsIgnoreCase(name)) {
                return value(matcher.group(2));
            }
        }
        return "";
    }

    private static String contextPrefix(
            String title,
            String source,
            String section,
            String kind,
            String speaker,
            String eventTime) {
        List<String> values = new ArrayList<>();
        add(values, "title", title);
        add(values, "source", source);
        add(values, "section", section);
        add(values, "kind", kind);
        add(values, "speaker", speaker);
        add(values, "time", eventTime);
        return String.join(" | ", values);
    }

    private static void add(List<String> output, String name, String value) {
        if (value != null && !value.isBlank()) {
            output.add(name + "=" + value.trim().replaceAll("\\s+", " "));
        }
    }

    private static String normalize(String text) {
        return value(text).replace("\r\n", "\n").replace('\r', '\n').trim();
    }

    private static String normalizeLabel(String value) {
        return value(value).trim().replaceAll("[_ ]+", " ");
    }

    private static String slug(String value) {
        return normalizeLabel(value).toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "_")
                .replaceAll("^_+|_+$", "");
    }

    private static String value(String value) {
        return value == null ? "" : value;
    }

    private record Heading(int start, int end, int level, String label) {
    }

    private record Boundary(int start, int end, String label) {
    }

    private record Range(int start, int end) {
    }

    private record TranscriptRange(int start, int end, String speaker, String time) {
    }

    private record Parent(
            String id,
            String kind,
            String sectionPath,
            String speaker,
            String threadId,
            String text,
            String eventTime) {

        Parent(
                String id,
                String kind,
                String sectionPath,
                String speaker,
                String threadId,
                String text) {
            this(id, kind, sectionPath, speaker, threadId, text, "");
        }
    }

    private record Leaf(
            String text,
            String kind,
            int start,
            int end,
            String speaker,
            String eventTime) {
    }
}
