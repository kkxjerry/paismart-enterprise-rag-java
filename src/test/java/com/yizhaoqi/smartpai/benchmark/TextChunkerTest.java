package com.yizhaoqi.smartpai.benchmark;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TextChunkerTest {

    @Test
    void prefersNaturalBoundariesAndCarriesOverlap() {
        String text = "alpha beta gamma. delta epsilon zeta. eta theta iota.";

        List<String> chunks = TextChunker.chunk(text, 32, 6);

        assertThat(chunks).hasSizeGreaterThan(1);
        assertThat(chunks.get(0)).doesNotEndWith(" ");
        // One overlap character is the trimmed boundary space.
        String overlap = chunks.get(0).substring(chunks.get(0).length() - 5);
        assertThat(chunks.get(1)).startsWith(overlap);
    }

    @Test
    void normalizesWhitespaceAndRejectsInvalidSizes() {
        assertThat(TextChunker.chunk(" a  b\r\n\r\n\r\n c ", 100, 10))
                .containsExactly("a b\n\n c");
        assertThatThrownBy(() -> TextChunker.chunk("text", 10, 10))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
