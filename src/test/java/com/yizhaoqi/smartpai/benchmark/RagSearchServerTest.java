package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.Headers;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class RagSearchServerTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void stripsOnlyServerArgumentsBeforeParsingRetrievalConfig() {
        String[] retrieval = RagSearchServer.stripServerArguments(new String[] {
                "--host", "127.0.0.1",
                "--port", "18090",
                "--questions", "questions.json",
                "--output", "summary.json",
                "--details-output", "details.jsonl",
                "--evidence-enabled", "true",
                "--index", "rag-index"
        });

        assertThat(retrieval).containsExactly(
                "--questions", "questions.json",
                "--output", "summary.json",
                "--details-output", "details.jsonl",
                "--evidence-enabled", "true",
                "--index", "rag-index");
    }

    @Test
    void onlineConfigDoesNotRequireBenchmarkQuestionOrOutputArguments() {
        EnterpriseRagJavaBenchmark.Config config = EnterpriseRagJavaBenchmark.Config.parseOnline(new String[] {
                "--index", "rag-index",
                "--embedding-model", "test-model",
                "--embedding-dimension", "2",
                "--evidence-enabled", "true"
        });

        assertThat(config.index()).isEqualTo("rag-index");
        assertThat(config.evidenceEnabled()).isTrue();
        assertThat(config.questions().toString()).contains("paismart-rag-online-unused");
        assertThat(config.output().toString()).contains("paismart-rag-online-unused");
    }

    @Test
    void productionAclFilterIsTenantScopedFailClosedAndDeletionAware() {
        SearchPrincipal principal = new SearchPrincipal(
                "tenant-a",
                List.of("group-b", "group-a", "group-a"),
                List.of("confidential", "internal"));

        JsonNode filter = EnterpriseRagJavaBenchmark.productionAclFilter(
                principal,
                List.of("jira", "confluence"));
        String json = filter.toString();

        assertThat(json).contains("tenant-a", "group-a", "group-b");
        assertThat(json).contains("jira", "confluence", "classification");
        assertThat(json).contains("allowedGroupIds", "deniedGroupIds", "deletedAt");
        assertThat(json).contains("minimum_should_match");
    }

    @Test
    void principalNormalizesAclIdentity() {
        SearchPrincipal principal = new SearchPrincipal(
                " tenant-a ",
                List.of(" group-b ", "group-a", "group-a", ""),
                List.of("internal", "internal"));

        assertThat(principal.tenantId()).isEqualTo("tenant-a");
        assertThat(principal.groupIds()).containsExactly("group-a", "group-b");
        assertThat(principal.classifications()).containsExactly("internal");
    }

    @Test
    void principalRequiresExplicitAuthorizedClassifications() {
        assertThatThrownBy(() -> new SearchPrincipal(
                "tenant-a",
                List.of("group-a"),
                List.of()))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("classification");
    }

    @Test
    void authenticationIsFailClosedAndUnauthenticatedModeIsLoopbackOnly() {
        RagSearchServer.ServerArguments secured = RagSearchServer.ServerArguments.parse(new String[] {});
        assertThat(secured.cacheTtlSeconds()).isZero();
        assertThatThrownBy(() -> RagSearchServer.validateAuthentication(secured, ""))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("API key");

        RagSearchServer.ServerArguments local = RagSearchServer.ServerArguments.parse(new String[] {
                "--allow-unauthenticated-loopback", "true"
        });
        RagSearchServer.validateAuthentication(local, "");

        RagSearchServer.ServerArguments exposed = RagSearchServer.ServerArguments.parse(new String[] {
                "--host", "0.0.0.0",
                "--allow-unauthenticated-loopback", "true"
        });
        assertThatThrownBy(() -> RagSearchServer.validateAuthentication(exposed, ""))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("loopback");
    }

    @Test
    void principalComesFromTrustedHeadersUnlessBodyModeIsExplicit() throws Exception {
        Headers headers = new Headers();
        headers.add("X-RAG-Tenant-Id", "tenant-header");
        headers.add("X-RAG-Group-Ids", "group-b, group-a");
        headers.add("X-RAG-Classifications", "internal, confidential");
        headers.add("X-RAG-Source-Types", "jira, slack");
        JsonNode request = MAPPER.readTree("""
                {
                  "principal": {
                    "tenant_id": "tenant-forged",
                    "group_ids": ["admin"],
                    "classifications": ["secret"]
                  },
                  "source_types": ["github"]
                }
                """);

        RagSearchServer.PrincipalAssertion trusted = RagSearchServer.principalAssertion(
                headers, request, false);
        assertThat(trusted.principal().tenantId()).isEqualTo("tenant-header");
        assertThat(trusted.principal().groupIds()).containsExactly("group-a", "group-b");
        assertThat(trusted.principal().classifications()).containsExactly("confidential", "internal");
        assertThat(trusted.sourceTypes()).containsExactly("jira", "slack");

        RagSearchServer.PrincipalAssertion development = RagSearchServer.principalAssertion(
                headers, request, true);
        assertThat(development.principal().tenantId()).isEqualTo("tenant-forged");
        assertThat(development.sourceTypes()).containsExactly("github");
    }

    @Test
    void cacheKeyChangesWithAclIdentity() {
        RagSearchServer.CacheKey first = new RagSearchServer.CacheKey(
                "question", "tenant-a", List.of("group-a"), List.of(), List.of("jira"), "rag", 20);
        RagSearchServer.CacheKey second = new RagSearchServer.CacheKey(
                "question", "tenant-a", List.of("group-b"), List.of(), List.of("jira"), "rag", 20);

        assertThat(first).isNotEqualTo(second);
    }

    @Test
    void validatesOnlineAclAndLifecycleFields() throws Exception {
        JsonNode valid = MAPPER.readTree("""
                {
                  "properties": {
                    "tenantId": {"type": "keyword"},
                    "sourceType": {"type": "keyword"},
                    "classification": {"type": "keyword"},
                    "allowedGroupIds": {"type": "keyword"},
                    "deniedGroupIds": {"type": "keyword"},
                    "aclHash": {"type": "keyword"},
                    "documentHash": {"type": "keyword"},
                    "contentHash": {"type": "keyword"},
                    "deletedAt": {"type": "date"}
                  }
                }
                """);

        RagSearchServer.validateOnlineMappings(valid);
        ((com.fasterxml.jackson.databind.node.ObjectNode) valid.path("properties")).remove("aclHash");
        assertThatThrownBy(() -> RagSearchServer.validateOnlineMappings(valid))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("aclHash");
    }

    @Test
    void rejectsNegativeCacheTtl() {
        assertThatThrownBy(() -> RagSearchServer.ServerArguments.parse(new String[] {
                "--cache-ttl-seconds", "-1"
        }))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cache-ttl-seconds");
    }
}
