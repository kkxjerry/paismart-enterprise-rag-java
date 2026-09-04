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
            case "sync" -> EnterpriseRagSynchronizer.main(commandArgs);
            case "lifecycle" -> IndexLifecycleCommand.main(commandArgs);
            case "serve-search" -> RagSearchServer.main(commandArgs);
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
                  import        Full chunk/embed/bulk import with fail-closed ACL handling
                  sync          Incremental sync: skip unchanged, update ACL, remove stale/missing chunks
                  lifecycle     Inspect or atomically promote a blue-green index alias
                  serve-search  Authenticated tenant/ACL-aware online retrieval API
                  evaluate      Run retrieval and optional bounded Java EvidenceBuilder evaluation

                Reproducible evaluation:
                  java -jar target/paismart-enterprise-rag.jar evaluate \\
                    --config config/experiments/sample-evidence-v1.json

                Explicit CLI arguments override values loaded from --config. Secrets must come from
                environment variables or explicit runtime arguments; they are redacted from manifests.
                """);
    }
}
