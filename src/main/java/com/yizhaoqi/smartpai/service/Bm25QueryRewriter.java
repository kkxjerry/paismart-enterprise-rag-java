package com.yizhaoqi.smartpai.service;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Builds a compact lexical query without changing the original user query.
 */
public final class Bm25QueryRewriter {

    private static final Pattern TOKEN_PATTERN = Pattern.compile(
            "[a-z0-9_][a-z0-9_./:+#-]*|[\\u4e00-\\u9fff]",
            Pattern.CASE_INSENSITIVE);

    private static final Set<String> STOPWORDS = Set.of(
            "a", "about", "according", "after", "all", "an", "and", "are", "as", "at",
            "be", "before", "by", "can", "company", "did", "do", "does", "during", "for",
            "from", "has", "have", "how", "i", "in", "into", "is", "it", "its", "me",
            "new", "of", "on", "or", "our", "should", "stated", "that", "the", "their",
            "they", "this", "to", "was", "we", "were", "what", "when", "where", "which",
            "who", "why", "with");

    private Bm25QueryRewriter() {
    }

    public static String keywordQuery(String query) {
        if (query == null || query.isBlank()) {
            return "";
        }
        String normalized = query.replaceAll("\\s+", " ").trim();
        Matcher matcher = TOKEN_PATTERN.matcher(normalized.toLowerCase(Locale.ROOT));
        List<String> keywords = new ArrayList<>();
        while (matcher.find()) {
            String token = matcher.group();
            if (!STOPWORDS.contains(token)
                    && (token.length() > 1 || isDigit(token) || isHan(token))) {
                keywords.add(token);
            }
        }
        return keywords.isEmpty() ? normalized : String.join(" ", keywords);
    }

    public static boolean isDistinctVariant(String original, String rewritten) {
        if (original == null || rewritten == null || rewritten.isBlank()) {
            return false;
        }
        return !original.replaceAll("\\s+", " ").trim().equalsIgnoreCase(rewritten);
    }

    private static boolean isDigit(String token) {
        return token.chars().allMatch(Character::isDigit);
    }

    private static boolean isHan(String token) {
        return token.codePoints().anyMatch(codePoint ->
                Character.UnicodeScript.of(codePoint) == Character.UnicodeScript.HAN);
    }
}
