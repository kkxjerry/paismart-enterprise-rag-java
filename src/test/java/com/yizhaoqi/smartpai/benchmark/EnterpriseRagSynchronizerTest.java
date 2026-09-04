package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class EnterpriseRagSynchronizerTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void distinguishesUnchangedMetadataOnlyAndContentChanges() throws Exception {
        EnterpriseRagImporter.Config config = EnterpriseRagImporter.Config.parse(new String[] {
                "--docs", "docs.jsonl",
                "--acl-docs", "acl.jsonl",
                "--index", "adaptive-v1",
                "--embedding-model", "test-embedding",
                "--chunking-strategy", "source-aware"
        });
        ObjectNode document = MAPPER.createObjectNode();
        document.put("doc_id", "doc-1");
        document.put("title", "Runbook");
        document.put("text", "Rollback to version v1.2.");
        document.put("document_version", "v1");
        document.put("source_updated_at", "2026-09-03T00:00:00Z");
        EnterpriseRagImporter.AclDocument acl = new EnterpriseRagImporter.AclDocument(
                "tenant-a", "internal", List.of("group-a"), List.of());

        EnterpriseRagSynchronizer.CurrentState current = EnterpriseRagSynchronizer.currentState(
                document, acl, config);
        EnterpriseRagSynchronizer.ExistingState identical = completeState(current);
        assertThat(identical.sameContent(current)).isTrue();
        assertThat(identical.sameMetadata(current)).isTrue();

        EnterpriseRagSynchronizer.ExistingState oldAcl = new EnterpriseRagSynchronizer.ExistingState(
                current.documentHash(),
                "old-acl",
                current.chunkingFingerprint(),
                current.modelVersion(),
                current.documentGeneration(),
                current.expectedChunkCount(),
                current.expectedChunkCount(),
                List.of(current.documentGeneration()),
                current.sourceType(),
                current.sourcePath(),
                current.sourceDataset(),
                current.documentVersion(),
                current.sourceUpdatedAt(),
                current.sourceRevision());
        assertThat(oldAcl.sameContent(current)).isTrue();
        assertThat(oldAcl.sameMetadata(current)).isFalse();

        EnterpriseRagSynchronizer.ExistingState oldContent = new EnterpriseRagSynchronizer.ExistingState(
                "old-document-hash",
                current.aclHash(),
                current.chunkingFingerprint(),
                current.modelVersion(),
                current.documentGeneration(),
                current.expectedChunkCount(),
                current.expectedChunkCount(),
                List.of(current.documentGeneration()),
                current.sourceType(),
                current.sourcePath(),
                current.sourceDataset(),
                current.documentVersion(),
                current.sourceUpdatedAt(),
                current.sourceRevision());
        assertThat(oldContent.sameContent(current)).isFalse();

        EnterpriseRagSynchronizer.ExistingState oldSourceType = new EnterpriseRagSynchronizer.ExistingState(
                current.documentHash(),
                current.aclHash(),
                current.chunkingFingerprint(),
                current.modelVersion(),
                current.documentGeneration(),
                current.expectedChunkCount(),
                current.expectedChunkCount(),
                List.of(current.documentGeneration()),
                "slack",
                current.sourcePath(),
                current.sourceDataset(),
                current.documentVersion(),
                current.sourceUpdatedAt(),
                current.sourceRevision());
        assertThat(oldSourceType.sameContent(current)).isFalse();

        EnterpriseRagSynchronizer.ExistingState oldSourcePath = new EnterpriseRagSynchronizer.ExistingState(
                current.documentHash(),
                current.aclHash(),
                current.chunkingFingerprint(),
                current.modelVersion(),
                current.documentGeneration(),
                current.expectedChunkCount(),
                current.expectedChunkCount(),
                List.of(current.documentGeneration()),
                current.sourceType(),
                "old:path",
                current.sourceDataset(),
                current.documentVersion(),
                current.sourceUpdatedAt(),
                current.sourceRevision());
        assertThat(oldSourcePath.sameContent(current)).isTrue();
        assertThat(oldSourcePath.sameMetadata(current)).isFalse();
    }

    @Test
    void existingStateQueryReadsOnlyLifecycleFingerprints() {
        String json = EnterpriseRagSynchronizer.existingStateQuery(List.of("doc-1", "doc-2")).toString();

        assertThat(json).contains("benchmarkDocId", "documentHash", "aclHash");
        assertThat(json).contains("chunkingFingerprint", "modelVersion");
        assertThat(json).contains("collapse");
        assertThat(json).doesNotContain("textContent", "vector");
    }

    @Test
    void metadataOnlyUpdateChangesAclWithoutTouchingVectors() throws Exception {
        EnterpriseRagImporter.AclDocument acl = new EnterpriseRagImporter.AclDocument(
                "tenant-b", "confidential", List.of("team-red"), List.of("team-blocked"));
        EnterpriseRagSynchronizer.CurrentState current = new EnterpriseRagSynchronizer.CurrentState(
                "doc-hash",
                acl.hash(),
                "source-aware:1200:200",
                "test-model",
                "generation-hash",
                3,
                "jira",
                "jira:runbook",
                "dataset",
                "v2",
                "2026-09-03T00:00:00Z",
                "revision-22",
                acl);
        ObjectNode body = EnterpriseRagSynchronizer.metadataUpdateBody(
                new EnterpriseRagSynchronizer.MetadataUpdate("doc-1", current));
        String json = body.toString();

        assertThat(json).contains("tenant-b", "team-red", "team-blocked", "aclHash");
        assertThat(json).contains("documentVersion", "sourceUpdatedAt", "sourceRevision");
        assertThat(json).doesNotContain("vector", "textContent", "documentHash");
    }

    @Test
    void staleChunkDeleteKeepsOnlyTheCurrentContentAndChunkingGeneration() {
        EnterpriseRagImporter.AclDocument acl = new EnterpriseRagImporter.AclDocument(
                "tenant-a", "internal", List.of(), List.of());
        EnterpriseRagSynchronizer.CurrentState current = new EnterpriseRagSynchronizer.CurrentState(
                "new-hash",
                acl.hash(),
                "source-aware:1200:200",
                "new-model",
                "new-generation",
                2,
                "confluence",
                "confluence:runbook",
                "dataset",
                "v2",
                "",
                "",
                acl);

        String json = EnterpriseRagSynchronizer.staleChunkDeleteBody("doc-1", current).toString();

        assertThat(json).contains("doc-1", "documentGeneration", "new-generation", "must_not");
        assertThat(json).doesNotContain("documentHash", "chunkingFingerprint", "modelVersion");
    }

    @Test
    void detectsIncompleteGenerationAndAllowsLegacyBackfillWithoutEmbedding() throws Exception {
        EnterpriseRagImporter.Config config = EnterpriseRagImporter.Config.parse(new String[] {
                "--docs", "docs.jsonl",
                "--acl-docs", "acl.jsonl",
                "--index", "adaptive-v1",
                "--embedding-model", "test-embedding",
                "--chunking-strategy", "source-aware"
        });
        ObjectNode document = MAPPER.createObjectNode();
        document.put("doc_id", "doc-1");
        document.put("title", "Incident runbook");
        document.put("text", "# Cause\nA stale route caused the incident.\n\n# Resolution\nRollback restored traffic.");
        document.put("source_type", "confluence");
        EnterpriseRagImporter.AclDocument acl = new EnterpriseRagImporter.AclDocument(
                "tenant-a", "internal", List.of("group-a"), List.of());
        EnterpriseRagSynchronizer.CurrentState current = EnterpriseRagSynchronizer.currentState(
                document, acl, config);

        EnterpriseRagSynchronizer.ExistingState partialWrite = new EnterpriseRagSynchronizer.ExistingState(
                current.documentHash(),
                current.aclHash(),
                current.chunkingFingerprint(),
                current.modelVersion(),
                current.documentGeneration(),
                current.expectedChunkCount(),
                Math.max(0, current.expectedChunkCount() - 1),
                List.of(current.documentGeneration()),
                current.sourceType(),
                current.sourcePath(),
                current.sourceDataset(),
                current.documentVersion(),
                current.sourceUpdatedAt(),
                current.sourceRevision());
        assertThat(partialWrite.sameContent(current)).isFalse();
        assertThat(partialWrite.canBackfillGeneration(current)).isFalse();

        EnterpriseRagSynchronizer.ExistingState mixedGenerations = new EnterpriseRagSynchronizer.ExistingState(
                current.documentHash(),
                current.aclHash(),
                current.chunkingFingerprint(),
                current.modelVersion(),
                current.documentGeneration(),
                current.expectedChunkCount(),
                current.expectedChunkCount(),
                List.of(current.documentGeneration(), "stale-generation"),
                current.sourceType(),
                current.sourcePath(),
                current.sourceDataset(),
                current.documentVersion(),
                current.sourceUpdatedAt(),
                current.sourceRevision());
        assertThat(mixedGenerations.sameContent(current)).isFalse();

        EnterpriseRagSynchronizer.ExistingState legacyComplete = new EnterpriseRagSynchronizer.ExistingState(
                current.documentHash(),
                current.aclHash(),
                current.chunkingFingerprint(),
                current.modelVersion(),
                "",
                0,
                current.expectedChunkCount(),
                List.of(),
                current.sourceType(),
                current.sourcePath(),
                current.sourceDataset(),
                current.documentVersion(),
                current.sourceUpdatedAt(),
                current.sourceRevision());
        assertThat(legacyComplete.sameContent(current)).isFalse();
        assertThat(legacyComplete.canBackfillGeneration(current)).isTrue();
    }

    private static EnterpriseRagSynchronizer.ExistingState completeState(
            EnterpriseRagSynchronizer.CurrentState current) {
        return new EnterpriseRagSynchronizer.ExistingState(
                current.documentHash(),
                current.aclHash(),
                current.chunkingFingerprint(),
                current.modelVersion(),
                current.documentGeneration(),
                current.expectedChunkCount(),
                current.expectedChunkCount(),
                List.of(current.documentGeneration()),
                current.sourceType(),
                current.sourcePath(),
                current.sourceDataset(),
                current.documentVersion(),
                current.sourceUpdatedAt(),
                current.sourceRevision());
    }

    @Test
    void deletionScopeRequiresExplicitManagedTenants() {
        assertThat(EnterpriseRagSynchronizer.managedTenantSet(" tenant-b,tenant-a,tenant-a "))
                .containsExactlyInAnyOrder("tenant-a", "tenant-b");
        assertThatThrownBy(() -> EnterpriseRagSynchronizer.validateDeletionScope(
                true,
                Set.of(),
                Set.of("tenant-a")))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("managed-tenants");
        assertThatThrownBy(() -> EnterpriseRagSynchronizer.validateDeletionScope(
                true,
                Set.of("tenant-a"),
                Set.of("tenant-a", "tenant-b")))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("tenant-b");
        EnterpriseRagSynchronizer.validateDeletionScope(
                true,
                Set.of("tenant-a", "tenant-b"),
                Set.of("tenant-a"));
    }

    @Test
    void chunkingFingerprintChangesWhenStrategyOrWindowChanges() {
        EnterpriseRagImporter.Config fixed = EnterpriseRagImporter.Config.parse(new String[] {
                "--docs", "docs.jsonl", "--acl-docs", "acl.jsonl", "--index", "a"
        });
        EnterpriseRagImporter.Config structured = EnterpriseRagImporter.Config.parse(new String[] {
                "--docs", "docs.jsonl", "--acl-docs", "acl.jsonl", "--index", "b",
                "--chunking-strategy", "source-aware", "--chunk-size", "800"
        });

        assertThat(EnterpriseRagSynchronizer.chunkingFingerprint(fixed)).isEqualTo("fixed:1200:200");
        assertThat(EnterpriseRagSynchronizer.chunkingFingerprint(structured))
                .isEqualTo("source-aware[*]:800:200");
    }
}
