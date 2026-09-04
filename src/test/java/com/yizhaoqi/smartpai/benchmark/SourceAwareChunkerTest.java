package com.yizhaoqi.smartpai.benchmark;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class SourceAwareChunkerTest {

    @Test
    void keepsJiraSectionsSeparate() {
        String text = """
                Summary:
                Uploads fail for large files.

                Root Cause:
                The gateway limit was 10 MiB.

                Workaround:
                Split the upload into smaller parts.
                """;

        List<SourceAwareChunker.Segment> segments = SourceAwareChunker.chunk(
                "jira", "Upload incident", text, 200, 20);

        assertThat(segments).extracting(SourceAwareChunker.Segment::kind)
                .contains("issue_summary", "issue_root_cause", "issue_workaround");
        assertThat(segments).allSatisfy(segment -> assertThat(segment.text()).doesNotContain("Summary:"));
    }

    @Test
    void preservesSlackSpeakerAndTimestamp() {
        String text = """
                [2026-09-03 10:00] Alice: Roll back build v1.2.
                [2026-09-03 10:01] Bob: Confirmed rollback completed.
                """;

        List<SourceAwareChunker.Segment> segments = SourceAwareChunker.chunk(
                "slack", "incident thread", text, 200, 20);

        assertThat(segments).hasSize(1);
        assertThat(segments.get(0).kind()).isEqualTo("conversation_window");
        assertThat(segments.get(0).speaker()).isEqualTo("Alice,Bob");
        assertThat(segments.get(0).eventTime()).contains("2026-09-03");
        assertThat(segments.get(0).threadId()).isEqualTo("thread");
        assertThat(segments.get(0).text()).contains("Alice:", "Bob:");
    }

    @Test
    void keepsMarkdownHeadingsAsSectionPaths() {
        String text = """
                # Runbook

                Intro text.

                ## Rollback

                Disable traffic and restore the prior release.
                """;

        List<SourceAwareChunker.Segment> segments = SourceAwareChunker.chunk(
                "confluence", "Runbook", text, 200, 20);

        assertThat(segments).extracting(SourceAwareChunker.Segment::sectionPath)
                .contains("Runbook", "Rollback");
    }

    @Test
    void coalescesSmallDriveParagraphsInsteadOfCreatingOneChunkPerParagraph() {
        String text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.";

        List<SourceAwareChunker.Segment> segments = SourceAwareChunker.chunk(
                "google_drive", "Notes", text, 200, 20);

        assertThat(segments).hasSize(1);
        assertThat(segments.get(0).text()).contains("Paragraph one", "Paragraph three");
    }

    @Test
    void importerCanRestrictSourceAwareChunkingToSelectedSources() {
        EnterpriseRagImporter.Config config = EnterpriseRagImporter.Config.parse(new String[] {
                "--docs", "docs.jsonl",
                "--acl-docs", "acl.jsonl",
                "--index", "selective",
                "--chunking-strategy", "source-aware",
                "--source-aware-types", "slack,linear"
        });
        List<SourceAwareChunker.Segment> slack = EnterpriseRagImporter.segmentsForDocument(
                config,
                "slack",
                "thread",
                "[10:00] Alice: Roll back.\n[10:01] Bob: Done.");
        List<SourceAwareChunker.Segment> confluence = EnterpriseRagImporter.segmentsForDocument(
                config,
                "confluence",
                "runbook",
                "# Cause\nStale route.\n\n# Fix\nRollback.");

        assertThat(slack).extracting(SourceAwareChunker.Segment::kind)
                .contains("conversation_window");
        assertThat(confluence).extracting(SourceAwareChunker.Segment::kind)
                .containsOnly("body");
        assertThat(EnterpriseRagImporter.chunkingFingerprint(config))
                .isEqualTo("source-aware[linear,slack]:1200:200");
    }

    @Test
    void fallsBackForUnknownSources() {
        List<SourceAwareChunker.Segment> segments = SourceAwareChunker.chunk(
                "unknown", "Title", "A short body.", 100, 10);

        assertThat(segments).singleElement().satisfies(segment -> {
            assertThat(segment.kind()).isEqualTo("body");
            assertThat(segment.sectionPath()).isEqualTo("Title");
        });
    }
}
