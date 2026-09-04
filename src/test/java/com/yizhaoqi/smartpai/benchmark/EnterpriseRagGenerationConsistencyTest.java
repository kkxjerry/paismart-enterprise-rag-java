package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

class EnterpriseRagGenerationConsistencyTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void acceptsOnlyOneCompleteCurrentGeneration() {
        EnterpriseRagSynchronizer.CurrentState current = current(3);
        EnterpriseRagSynchronizer.ExistingState existing = existing(
                "generation-current",
                3,
                3,
                Set.of("generation-current"));

        assertThat(existing.sameContent(current)).isTrue();
        assertThat(existing.completeGeneration(current)).isTrue();
        assertThat(existing.needsGenerationBackfill()).isFalse();
    }

    @Test
    void rejectsPartiallyWrittenCurrentGeneration() {
        EnterpriseRagSynchronizer.CurrentState current = current(3);
        EnterpriseRagSynchronizer.ExistingState existing = existing(
                "generation-current",
                3,
                1,
                Set.of("generation-current"));

        assertThat(existing.sameContent(current)).isTrue();
        assertThat(existing.completeGeneration(current)).isFalse();
    }

    @Test
    void rejectsMixedOldAndCurrentGenerationsAfterInterruptedCleanup() {
        EnterpriseRagSynchronizer.CurrentState current = current(3);
        EnterpriseRagSynchronizer.ExistingState existing = existing(
                "generation-current",
                3,
                5,
                Set.of("generation-current", "generation-old"));

        assertThat(existing.completeGeneration(current)).isFalse();
    }

    @Test
    void permitsExactLegacyGenerationOnlyForMetadataBackfill() {
        EnterpriseRagSynchronizer.CurrentState current = current(3);
        EnterpriseRagSynchronizer.ExistingState legacy = existing(
                "",
                0,
                3,
                Set.of());
        EnterpriseRagSynchronizer.ExistingState incompleteLegacy = existing(
                "",
                0,
                2,
                Set.of());

        assertThat(legacy.completeGeneration(current)).isTrue();
        assertThat(legacy.needsGenerationBackfill()).isTrue();
        assertThat(incompleteLegacy.completeGeneration(current)).isFalse();
    }

    @Test
    void parsesPerDocumentObservedGenerationCounts() throws Exception {
        Map<String, EnterpriseRagSynchronizer.GenerationObservation> observations =
                EnterpriseRagSynchronizer.generationObservations(MAPPER.readTree("""
                        {
                          "aggregations": {
                            "by_document": {
                              "buckets": [
                                {
                                  "key": "doc-1",
                                  "doc_count": 3,
                                  "generations": {
                                    "buckets": [
                                      {"key": "generation-current", "doc_count": 2},
                                      {"key": "generation-old", "doc_count": 1}
                                    ]
                                  }
                                }
                              ]
                            }
                          }
                        }
                        """));

        assertThat(observations).containsOnlyKeys("doc-1");
        assertThat(observations.get("doc-1").chunkCount()).isEqualTo(3);
        assertThat(observations.get("doc-1").generations())
                .containsExactlyInAnyOrder("generation-current", "generation-old");
    }

    @Test
    void staleCleanupKeepsOnlyTheCurrentGeneration() {
        String body = EnterpriseRagSynchronizer.staleChunkDeleteBody("doc-1", current(3)).toString();

        assertThat(body).contains("documentGeneration", "generation-current", "must_not");
        assertThat(body).doesNotContain("documentHash", "chunkingFingerprint", "modelVersion");
    }

    private static EnterpriseRagSynchronizer.CurrentState current(int chunkCount) {
        EnterpriseRagImporter.AclDocument acl = new EnterpriseRagImporter.AclDocument(
                "tenant-a",
                "internal",
                List.of("team-a"),
                List.of());
        return new EnterpriseRagSynchronizer.CurrentState(
                "document-hash",
                acl.hash(),
                "source-aware:1200:200",
                "test-model",
                "generation-current",
                chunkCount,
                "confluence",
                "confluence:runbook",
                "dataset",
                "v2",
                "2026-09-03T00:00:00Z",
                "revision-2",
                acl);
    }

    private static EnterpriseRagSynchronizer.ExistingState existing(
            String generation,
            int expectedCount,
            long observedCount,
            Set<String> observedGenerations) {
        EnterpriseRagSynchronizer.CurrentState current = current(3);
        return new EnterpriseRagSynchronizer.ExistingState(
                current.documentHash(),
                current.aclHash(),
                current.chunkingFingerprint(),
                current.modelVersion(),
                generation,
                expectedCount,
                observedCount,
                observedGenerations,
                current.sourceType(),
                current.sourcePath(),
                current.sourceDataset(),
                current.documentVersion(),
                current.sourceUpdatedAt(),
                current.sourceRevision());
    }
}
