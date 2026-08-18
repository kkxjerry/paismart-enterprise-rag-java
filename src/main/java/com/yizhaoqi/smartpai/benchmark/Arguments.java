package com.yizhaoqi.smartpai.benchmark;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

final class Arguments {

    private final Map<String, String> values;

    private Arguments(Map<String, String> values) {
        this.values = Map.copyOf(values);
    }

    static Arguments parse(String[] args) {
        Map<String, String> values = new LinkedHashMap<>();
        for (int index = 0; index < args.length; index += 2) {
            if (index + 1 >= args.length || !args[index].startsWith("--")) {
                throw new IllegalArgumentException("arguments must use --name value pairs");
            }
            String key = args[index].substring(2);
            if (key.isBlank() || values.put(key, args[index + 1]) != null) {
                throw new IllegalArgumentException("duplicate or empty argument: --" + key);
            }
        }
        return new Arguments(values);
    }

    String required(String key) {
        String value = values.get(key);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("missing --" + key);
        }
        return value;
    }

    Path requiredPath(String key) {
        return Path.of(required(key));
    }

    String string(String key, String defaultValue) {
        return values.getOrDefault(key, defaultValue);
    }

    Path path(String key, String defaultValue) {
        return Path.of(string(key, defaultValue));
    }

    int positiveInt(String key, int defaultValue) {
        int value = Integer.parseInt(string(key, Integer.toString(defaultValue)));
        if (value <= 0) {
            throw new IllegalArgumentException("--" + key + " must be positive");
        }
        return value;
    }

    int nonNegativeInt(String key, int defaultValue) {
        int value = Integer.parseInt(string(key, Integer.toString(defaultValue)));
        if (value < 0) {
            throw new IllegalArgumentException("--" + key + " must not be negative");
        }
        return value;
    }

    boolean bool(String key, boolean defaultValue) {
        String value = string(key, Boolean.toString(defaultValue)).toLowerCase();
        if (!Set.of("true", "false").contains(value)) {
            throw new IllegalArgumentException("--" + key + " must be true or false");
        }
        return Boolean.parseBoolean(value);
    }

    String oneOf(String key, String defaultValue, Set<String> allowed) {
        String value = string(key, defaultValue).toLowerCase();
        if (!allowed.contains(value)) {
            throw new IllegalArgumentException("--" + key + " must be one of " + allowed);
        }
        return value;
    }
}
