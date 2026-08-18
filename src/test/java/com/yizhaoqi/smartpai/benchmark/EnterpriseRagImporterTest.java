package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class EnterpriseRagImporterTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void checkpointSignatureTreatsEquivalentJsonNumbersAsEqual() throws Exception {
        Map<String, Object> signature = new LinkedHashMap<>();
        signature.put("docs_size", 404L);
        signature.put("docs_mtime_ms", 1_787_000_000_000L);
        signature.put("index", "benchmark_v1");
        boolean matches = EnterpriseRagImporter.signatureMatches(
                MAPPER.readTree("{\"docs_size\":404,\"docs_mtime_ms\":1787000000000,\"index\":\"benchmark_v1\"}"),
                signature);

        assertThat(matches).isTrue();
    }
}
