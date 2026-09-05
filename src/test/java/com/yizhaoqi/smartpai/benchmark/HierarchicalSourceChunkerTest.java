package com.yizhaoqi.smartpai.benchmark;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class HierarchicalSourceChunkerTest {

    @Test
    void confluenceLeavesKeepHeadingParentAndExactRawSlice() {
        String text = """
                # Runbook
                Intro.

                ## Rollback
                - Take a snapshot.
                - Restore usually takes tens of minutes.
                - Exact time depends on database size.
                """;

        List<SourceAwareChunker.Segment> values = HierarchicalSourceChunker.chunk(
                "confluence", "Private Upgrade", text, 120, 10);

        assertThat(values).isNotEmpty();
        SourceAwareChunker.Segment restore = values.stream()
                .filter(value -> value.text().contains("tens of minutes"))
                .findFirst()
                .orElseThrow();
        assertThat(restore.sectionPath()).isEqualTo("Runbook / Rollback");
        assertThat(restore.parentText()).contains("Take a snapshot", "database size");
        assertThat(restore.text()).isEqualTo(
                restore.parentText().substring(restore.parentStart(), restore.parentEnd()));
        assertThat(restore.contextPrefix()).contains(
                "title=Private Upgrade", "source=confluence", "section=Runbook / Rollback");
    }

    @Test
    void gmailUsesOneMessageAsParentInsteadOfMixingThreadReplies() {
        String text = """
                From: Customer <customer@example.com>
                Date: Tue, Jun 3, 2025 at 9:12 AM
                Subject: Upgrade question

                How long does rollback take?

                From: Support <support@example.com>
                Date: Tue, Jun 3, 2025 at 11:26 AM
                Subject: Re: Upgrade question

                With snapshots, restoration is typically tens of minutes.
                Exact timing depends on the snapshot mechanism and database size.
                """;

        List<SourceAwareChunker.Segment> values = HierarchicalSourceChunker.chunk(
                "gmail", "Upgrade thread", text, 180, 20);

        SourceAwareChunker.Segment answer = values.stream()
                .filter(value -> value.text().contains("typically tens of minutes"))
                .findFirst()
                .orElseThrow();
        assertThat(answer.parentId()).isEqualTo("parent-2");
        assertThat(answer.parentText()).contains("From: Support", "database size");
        assertThat(answer.parentText()).doesNotContain("From: Customer");
        assertThat(answer.contextPrefix()).contains("source=gmail", "section=Re: Upgrade question");
    }

    @Test
    void firefliesSeparatesSummaryAndTranscriptAndPreservesSpeakerTime() {
        String text = """
                summary:
                The team agreed to run a pilot next week.

                transcript:
                [00:10] Alice: The p95 target is 120 ms.
                [00:20] Bob: Upload samples by Friday.

                next_steps:
                - Alice owns the benchmark.
                - Bob uploads samples.
                """;

        List<SourceAwareChunker.Segment> values = HierarchicalSourceChunker.chunk(
                "fireflies", "Pilot review", text, 150, 10);

        SourceAwareChunker.Segment turn = values.stream()
                .filter(value -> value.text().contains("p95 target"))
                .findFirst()
                .orElseThrow();
        assertThat(turn.kind()).isEqualTo("transcript_turn");
        assertThat(turn.speaker()).isEqualTo("Alice");
        assertThat(turn.eventTime()).isEqualTo("00:10");
        assertThat(turn.sectionPath()).isEqualTo("transcript");
        assertThat(values).anySatisfy(value -> {
            assertThat(value.sectionPath()).isEqualTo("next steps");
            assertThat(value.parentText()).contains("Alice owns", "Bob uploads");
        });
    }

    @Test
    void importerParentChildStrategyIsOptInAndFingerprintIsDistinct() {
        EnterpriseRagImporter.Config config = EnterpriseRagImporter.Config.parse(new String[] {
                "--docs", "docs.jsonl",
                "--acl-docs", "acl.jsonl",
                "--index", "parent-child",
                "--chunking-strategy", "parent-child",
                "--source-aware-types", "confluence,fireflies,gmail",
                "--chunk-size", "500",
                "--chunk-overlap", "50"
        });

        List<SourceAwareChunker.Segment> gmail = EnterpriseRagImporter.segmentsForDocument(
                config,
                "gmail",
                "thread",
                "From: A <a@example.com>\nSubject: Test\n\nA message body.");
        List<SourceAwareChunker.Segment> slack = EnterpriseRagImporter.segmentsForDocument(
                config,
                "slack",
                "thread",
                "[10:00] Alice: A message body.");

        assertThat(gmail).allSatisfy(value -> assertThat(value.parentId()).isNotBlank());
        assertThat(slack).allSatisfy(value -> assertThat(value.parentId()).isBlank());
        assertThat(EnterpriseRagImporter.chunkingFingerprint(config))
                .isEqualTo("parent-child[confluence,fireflies,gmail]:500:50:context-prefix=true");
    }

    @Test
    void contextPrefixCanBeDisabledForAnIsolatedE4Experiment() {
        EnterpriseRagImporter.Config config = EnterpriseRagImporter.Config.parse(new String[] {
                "--docs", "docs.jsonl",
                "--acl-docs", "acl.jsonl",
                "--index", "parent-child-no-prefix",
                "--chunking-strategy", "parent-child",
                "--source-aware-types", "gmail",
                "--context-prefix-enabled", "false"
        });

        assertThat(config.contextPrefixEnabled()).isFalse();
        assertThat(EnterpriseRagImporter.chunkingFingerprint(config))
                .isEqualTo("parent-child[gmail]:1200:200:context-prefix=false");
    }

    @Test
    void buildChunksStoresParentMetadataWithoutChangingRawLeaf() {
        EnterpriseRagImporter.Config config = EnterpriseRagImporter.Config.parse(new String[] {
                "--docs", "docs.jsonl",
                "--acl-docs", "acl.jsonl",
                "--index", "parent-child",
                "--chunking-strategy", "parent-child",
                "--source-aware-types", "gmail",
                "--chunk-size", "500",
                "--chunk-overlap", "50",
                "--fail-on-missing-acl", "false"
        });
        var document = new com.fasterxml.jackson.databind.ObjectMapper().createObjectNode()
                .put("doc_id", "doc-1")
                .put("title", "Thread")
                .put("source_type", "gmail")
                .put("text", "From: Support <s@example.com>\nSubject: Restore\n\nRestore takes 20 minutes.");

        List<EnterpriseRagImporter.Chunk> chunks = EnterpriseRagImporter.buildChunks(
                config, List.of(document), java.util.Map.of());

        assertThat(chunks).hasSize(2);
        EnterpriseRagImporter.Chunk answer = chunks.stream()
                .filter(chunk -> chunk.source().path("textContent").asText().contains("20 minutes"))
                .findFirst()
                .orElseThrow();
        var source = answer.source();
        assertThat(source.path("parentId").asText()).startsWith("doc-1:parent-");
        assertThat(source.path("parentText").asText()).contains("Restore takes 20 minutes");
        assertThat(source.path("contextPrefix").asText()).contains("source=gmail");
        assertThat(source.path("textContent").asText()).isEqualTo(
                source.path("parentText").asText().substring(
                        source.path("parentStart").asInt(), source.path("parentEnd").asInt()));
    }
}
