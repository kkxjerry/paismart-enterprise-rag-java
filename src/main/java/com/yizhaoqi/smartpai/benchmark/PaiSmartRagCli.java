package com.yizhaoqi.smartpai.benchmark;

import java.util.Arrays;

public final class PaiSmartRagCli {

    private PaiSmartRagCli() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0 || "help".equals(args[0]) || "--help".equals(args[0])) {
            printHelp();
            return;
        }
        String[] commandArgs = Arrays.copyOfRange(args, 1, args.length);
        switch (args[0]) {
            case "create-index" -> ElasticsearchIndexCommand.main(commandArgs);
            case "import" -> EnterpriseRagImporter.main(commandArgs);
            case "evaluate" -> EnterpriseRagJavaBenchmark.main(commandArgs);
            default -> throw new IllegalArgumentException("unknown command: " + args[0]);
        }
    }

    private static void printHelp() {
        System.out.println("""
                PaiSmart EnterpriseRAG Java Benchmark

                Usage:
                  java -jar target/paismart-enterprise-rag.jar <command> [--name value ...]

                Commands:
                  create-index  Create an isolated Elasticsearch benchmark index
                  import        Chunk, embed, and bulk-index EnterpriseRAG JSONL documents
                  evaluate      Run Dense, BM25, or weighted multi-route Hybrid evaluation

                Run the command examples in README.md for complete reproducible arguments.
                """);
    }
}
