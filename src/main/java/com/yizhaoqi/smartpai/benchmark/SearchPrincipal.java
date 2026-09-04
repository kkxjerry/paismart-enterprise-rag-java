package com.yizhaoqi.smartpai.benchmark;

import java.util.List;

/** Authenticated identity used by the online retrieval API. */
record SearchPrincipal(
        String tenantId,
        List<String> groupIds,
        List<String> classifications) {

    SearchPrincipal {
        tenantId = tenantId == null ? "" : tenantId.trim();
        groupIds = groupIds == null ? List.of() : groupIds.stream()
                .map(String::trim)
                .filter(value -> !value.isBlank())
                .distinct()
                .sorted()
                .toList();
        classifications = classifications == null ? List.of() : classifications.stream()
                .map(String::trim)
                .filter(value -> !value.isBlank())
                .distinct()
                .sorted()
                .toList();
        if (tenantId.isBlank()) {
            throw new IllegalArgumentException("tenant_id is required");
        }
        if (classifications.isEmpty()) {
            throw new IllegalArgumentException(
                    "at least one authorized classification is required");
        }
    }
}
