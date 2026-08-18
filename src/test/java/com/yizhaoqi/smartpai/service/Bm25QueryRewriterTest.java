package com.yizhaoqi.smartpai.service;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class Bm25QueryRewriterTest {

    @Test
    void reproducesTheEnterpriseRagKeywordVariant() {
        String query = "What is the company policy for how long contractor access should last "
                + "by default before it expires, according to the access and permissions playbook?";

        assertThat(Bm25QueryRewriter.keywordQuery(query))
                .isEqualTo("policy long contractor access last default expires access permissions playbook");
    }

    @Test
    void retainsNumbersIdentifiersAndChineseTerms() {
        assertThat(Bm25QueryRewriter.keywordQuery("What is API_v2 p99 for 上海节点 48-hour runs?"))
                .isEqualTo("api_v2 p99 上 海 节 点 48-hour runs");
    }

    @Test
    void fallsBackToOriginalWhenNoKeywordRemains() {
        assertThat(Bm25QueryRewriter.keywordQuery("what is the"))
                .isEqualTo("what is the");
        assertThat(Bm25QueryRewriter.isDistinctVariant("what is the", "what is the")).isFalse();
    }
}
