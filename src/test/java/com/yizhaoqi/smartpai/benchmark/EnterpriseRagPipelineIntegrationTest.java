package com.yizhaoqi.smartpai.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;

class EnterpriseRagPipelineIntegrationTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @TempDir
    Path tempDir;

    @Test
    void runsConfigRetrievalEvidenceOutputsAndManifestEndToEnd() throws Exception {
        AtomicInteger indexMetadataRequests = new AtomicInteger();
        AtomicInteger embeddingRequests = new AtomicInteger();
        AtomicInteger retrievalRequests = new AtomicInteger();
        AtomicInteger evidenceRequests = new AtomicInteger();
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> handle(
                exchange,
                indexMetadataRequests,
                embeddingRequests,
                retrievalRequests,
                evidenceRequests));
        server.start();
        try {
            int port = server.getAddress().getPort();
            Path questions = tempDir.resolve("questions.json");
            Files.writeString(questions, """
                    {
                      "questions": [
                        {
                          "id": "q1",
                          "question": "What are the upload limits?",
                          "question_type": "basic",
                          "source_types": ["github"],
                          "expected_doc_ids": ["doc-1"],
                          "gold_answer": "10 MiB per file and 50 MiB per request.",
                          "answer_facts": [
                            "The default max_file_size is 10 MiB per file.",
                            "The default max_total_request_size is 50 MiB per request."
                          ]
                        }
                      ]
                    }
                    """);
            Path config = tempDir.resolve("experiment.json");
            Files.writeString(config, """
                    {
                      "schema_version": 1,
                      "name": "integration-evidence",
                      "base_dir": ".",
                      "arguments": {
                        "run-id": "integration-evidence",
                        "questions": "questions.json",
                        "output": "summary.json",
                        "details-output": "details.jsonl",
                        "manifest-output": "manifest.json",
                        "evidence-output": "contexts.jsonl",
                        "es-url": "http://127.0.0.1:%d",
                        "index": "test-index",
                        "embedding-url": "http://127.0.0.1:%d/v1/embeddings",
                        "embedding-model": "test-model",
                        "embedding-dimension": 2,
                        "retrieval-mode": "hybrid",
                        "retriever-k": 5,
                        "dense-chunk-candidates": 5,
                        "dense-num-candidates": 10,
                        "rrf-k": 10,
                        "dense-weight": 0.75,
                        "bm25-weight": 0.50,
                        "keyword-bm25-enabled": true,
                        "keyword-bm25-weight": 1.25,
                        "english-bm25-enabled": true,
                        "english-bm25-weight": 1.50,
                        "top-k": 5,
                        "evidence-enabled": true,
                        "evidence-top-documents": 1,
                        "evidence-candidate-chunks-per-document": 2,
                        "evidence-chunks-per-document": 2,
                        "evidence-token-budget": 80,
                        "evidence-per-document-token-budget": 80,
                        "evidence-max-chunk-tokens": 40,
                        "progress-every": 0
                      },
                      "inputs": {
                        "questions": "questions.json"
                      },
                      "metadata": {
                        "embedding": {
                          "model": "test-model",
                          "native_dimension": 2,
                          "stored_dimension": 2,
                          "dimension_strategy": "native"
                        }
                      }
                    }
                    """.formatted(port, port));

            ByteArrayOutputStream capturedOutput = new ByteArrayOutputStream();
            PrintStream originalOutput = System.out;
            try (PrintStream capture = new PrintStream(capturedOutput, true, StandardCharsets.UTF_8)) {
                System.setOut(capture);
                EnterpriseRagJavaBenchmark.main(new String[] {"--config", config.toString()});
            } finally {
                System.setOut(originalOutput);
            }
            assertThat(capturedOutput.toString(StandardCharsets.UTF_8))
                    .contains("\"run_id\" : \"integration-evidence\"");

            JsonNode summary = MAPPER.readTree(tempDir.resolve("summary.json").toFile());
            assertThat(summary.path("run_id").asText()).isEqualTo("integration-evidence");
            assertThat(summary.path("hit@10").asDouble()).isEqualTo(1.0d);
            assertThat(summary.path("evidence_fact_token_recall_avg").asDouble()).isEqualTo(1.0d);
            assertThat(summary.path("evidence_fact_coverage_avg").asDouble()).isEqualTo(1.0d);
            assertThat(summary.path("evidence_source_filter_violation_count").asInt()).isZero();
            assertThat(summary.path("document_ranking_changed_by_evidence_count").asInt()).isZero();

            JsonNode details = MAPPER.readTree(Files.readAllLines(tempDir.resolve("details.jsonl")).get(0));
            assertThat(details.path("ranked_documents").path(0).path("doc_id").asText()).isEqualTo("doc-1");
            assertThat(details.path("ranked_documents").path(0).path("route_contributions")).hasSize(4);
            assertThat(details.path("evidence").path("spans")).hasSize(2);

            JsonNode contexts = MAPPER.readTree(Files.readAllLines(tempDir.resolve("contexts.jsonl")).get(0));
            assertThat(contexts.path("context_mode").asText()).isEqualTo("java_evidence_builder");
            assertThat(contexts.path("contexts")).hasSize(2);
            assertThat(contexts.path("contexts").path(0).path("citation_id").asText()).isEqualTo("S1");
            assertThat(contexts.path("contexts").path(1).path("citation_id").asText()).isEqualTo("S2");
            assertThat(contexts.path("contexts").toString()).contains("10 MiB", "50 MiB");

            JsonNode manifest = MAPPER.readTree(tempDir.resolve("manifest.json").toFile());
            assertThat(manifest.path("status").asText()).isEqualTo("completed");
            assertThat(manifest.path("code").path("commit").asText()).isEqualTo("unknown");
            assertThat(manifest.path("code").path("dirty").isNull()).isTrue();
            assertThat(manifest.path("experiment").path("config_sha256").asText()).hasSize(64);
            assertThat(manifest.path("inputs").path("questions").path("sha256").asText()).hasSize(64);
            assertThat(manifest.path("index_metadata").path("vector_dimension").asInt()).isEqualTo(2);
            assertThat(manifest.path("summary").path("hit@10").asDouble()).isEqualTo(1.0d);
            assertThat(manifest.path("outputs").path("summary").path("sha256").asText()).hasSize(64);
            assertThat(manifest.path("outputs").path("details").path("sha256").asText()).hasSize(64);
            assertThat(manifest.path("outputs").path("evidence").path("sha256").asText()).hasSize(64);

            assertThat(indexMetadataRequests).hasValue(1);
            assertThat(embeddingRequests).hasValue(1);
            assertThat(retrievalRequests).hasValue(4);
            assertThat(evidenceRequests).hasValue(1);
        } finally {
            server.stop(0);
        }
    }

    private static void handle(
            HttpExchange exchange,
            AtomicInteger indexMetadataRequests,
            AtomicInteger embeddingRequests,
            AtomicInteger retrievalRequests,
            AtomicInteger evidenceRequests) throws IOException {
        String path = exchange.getRequestURI().getPath();
        if ("GET".equals(exchange.getRequestMethod()) && "/test-index".equals(path)) {
            indexMetadataRequests.incrementAndGet();
            respond(exchange, 200, indexMetadataResponse());
            return;
        }
        if ("POST".equals(exchange.getRequestMethod()) && "/v1/embeddings".equals(path)) {
            embeddingRequests.incrementAndGet();
            respond(exchange, 200, "{\"data\":[{\"index\":0,\"embedding\":[1.0,0.0]}]}");
            return;
        }
        if ("POST".equals(exchange.getRequestMethod()) && "/test-index/_search".equals(path)) {
            String requestBody = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            if (requestBody.contains("evidence_chunks")) {
                evidenceRequests.incrementAndGet();
                respond(exchange, 200, evidenceResponse());
            } else {
                retrievalRequests.incrementAndGet();
                respond(exchange, 200, retrievalResponse());
            }
            return;
        }
        respond(exchange, 404, "{\"error\":\"not found\"}");
    }

    private static void respond(HttpExchange exchange, int status, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(status, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.close();
    }

    private static String indexMetadataResponse() {
        return """
                {
                  "test-index": {
                    "settings": {
                      "index": {
                        "number_of_shards": "1",
                        "number_of_replicas": "0",
                        "similarity": {
                          "enterprise_bm25": {"type": "BM25", "k1": "2.2", "b": "1.0"}
                        }
                      }
                    },
                    "mappings": {
                      "_meta": {
                        "embedding_model": "test-model",
                        "embedding_dimension": 2
                      },
                      "properties": {
                        "vector": {"type": "dense_vector", "dims": 2, "similarity": "cosine"},
                        "textContent": {"type": "text", "analyzer": "standard"},
                        "documentVersion": {"type": "keyword"},
                        "documentHash": {"type": "keyword"},
                        "contentHash": {"type": "keyword"},
                        "sourceUpdatedAt": {"type": "date"}
                      }
                    }
                  }
                }
                """;
    }

    private static String retrievalResponse() {
        return """
                {
                  "hits": {
                    "hits": [
                      {
                        "_id": "doc-1:00001",
                        "_score": 10.0,
                        "_source": {
                          "benchmarkDocId": "doc-1",
                          "chunkId": 1,
                          "sourceType": "github",
                          "sourcePath": "github:upload-limits",
                          "title": "Upload limits",
                          "textContent": "The default max_file_size is 10 MiB per file.",
                          "classification": "internal",
                          "documentVersion": "v1",
                          "documentHash": "doc-hash",
                          "sourceUpdatedAt": "2026-08-18T00:00:00Z",
                          "contentHash": "chunk-hash-1",
                          "indexedAt": "2026-08-18T00:00:00Z"
                        }
                      }
                    ]
                  }
                }
                """;
    }

    private static String evidenceResponse() {
        return """
                {
                  "hits": {
                    "hits": [
                      {
                        "_id": "doc-1:00001",
                        "_score": 10.0,
                        "_source": {
                          "benchmarkDocId": "doc-1",
                          "chunkId": 1,
                          "sourceType": "github",
                          "sourcePath": "github:upload-limits",
                          "title": "Upload limits",
                          "textContent": "The default max_file_size is 10 MiB per file.",
                          "classification": "internal",
                          "documentVersion": "v1",
                          "documentHash": "doc-hash",
                          "sourceUpdatedAt": "2026-08-18T00:00:00Z",
                          "contentHash": "chunk-hash-1",
                          "indexedAt": "2026-08-18T00:00:00Z"
                        },
                        "inner_hits": {
                          "evidence_chunks": {
                            "hits": {
                              "hits": [
                                {
                                  "_id": "doc-1:00001",
                                  "_score": 9.0,
                                  "_source": {
                                    "benchmarkDocId": "doc-1",
                                    "chunkId": 1,
                                    "sourceType": "github",
                                    "sourcePath": "github:upload-limits",
                                    "title": "Upload limits",
                                    "textContent": "The default max_file_size is 10 MiB per file.",
                                    "classification": "internal",
                                    "documentVersion": "v1",
                                    "documentHash": "doc-hash",
                                    "sourceUpdatedAt": "2026-08-18T00:00:00Z",
                                    "contentHash": "chunk-hash-1",
                                    "indexedAt": "2026-08-18T00:00:00Z"
                                  }
                                },
                                {
                                  "_id": "doc-1:00002",
                                  "_score": 8.0,
                                  "_source": {
                                    "benchmarkDocId": "doc-1",
                                    "chunkId": 2,
                                    "sourceType": "github",
                                    "sourcePath": "github:upload-limits",
                                    "title": "Upload limits",
                                    "textContent": "The default max_total_request_size is 50 MiB per request.",
                                    "classification": "internal",
                                    "documentVersion": "v1",
                                    "documentHash": "doc-hash",
                                    "sourceUpdatedAt": "2026-08-18T00:00:00Z",
                                    "contentHash": "chunk-hash-2",
                                    "indexedAt": "2026-08-18T00:00:00Z"
                                  }
                                }
                              ]
                            }
                          }
                        }
                      }
                    ]
                  }
                }
                """;
    }
}
