# Generation 一致性测试修复与验收

验收完成时间：2026-09-05 04:49:05 UTC（洛杉矶 2026-09-04 21:49:05）。
仓库：/Users/zhouguichao/Downloads/minirag_demo/github/paismart-enterprise-rag-java
分支：develop；基线提交：209fc00。修改保留在 develop 工作区，尚未提交或推送。本次没有创建 worktree，也未删除其他会话留下的 worktree。

## 问题与改动

0a8e981 同时引入 EnterpriseRagSynchronizer 和 EnterpriseRagGenerationConsistencyTest，提交说明已明确记录该测试不能编译；不是 209fc00 引入的问题。现有 Git 记录不足以证明旧版测试曾在更早提交通过。

代码仅修改 src/test/java/com/yizhaoqi/smartpai/benchmark/EnterpriseRagGenerationConsistencyTest.java；生产代码和 pom.xml 未修改。

- 使用 sameContent(current) 验证内容和 generation 完整性，不再调用私有 completeGeneration。
- 使用 canBackfillGeneration(current) 验证旧索引回填资格；完整旧索引也不能被当作 unchanged，回填必须由同步参数显式允许。
- observedGenerations 使用当前构造器要求的 List<String>，移除不存在的 needsGenerationBackfill、GenerationObservation、generationObservations 引用。
- 通过同步 main 入口和随机端口的 127.0.0.1 HTTP 测试服务覆盖实际聚合解析，不增加废弃生产接口，不使用反射。
- 保留原有完整、部分写入、混合代际、旧数据回填、陈旧分片清理场景；补充元数据不一致、缺失聚合、默认禁止旧数据回填等检查。

目标测试共 14 个执行用例：7 个普通测试和 7 个参数化场景。HTTP 场景使用 dry-run，断言只发生 mapping/search 请求，不访问真实 Elasticsearch、模型或 A40。没有使用 Docker。

## JDK 17 验收

Maven 3.9.9；Amazon Corretto 17.0.20.1，macOS aarch64。

依次执行并全部成功：

```sh
export JAVA_HOME=/tmp/rag-jdk17.PpHMBF/amazon-corretto-17.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$PATH"
SETTINGS=/opt/homebrew/Cellar/maven/3.9.9/libexec/conf/settings.xml

mvn -v
mvn -B -ntp -s "$SETTINGS" -DskipTests test-compile
mvn -B -ntp -s "$SETTINGS" -Dmaven.test.redirectTestOutputToFile=true -Dtest=EnterpriseRagGenerationConsistencyTest test
mvn -B -ntp -s "$SETTINGS" -Dmaven.test.redirectTestOutputToFile=true clean verify
```

- test-compile：BUILD SUCCESS。
- 单类测试：14 tests，0 failures，0 errors，0 skipped。
- clean verify：重新编译 18 个生产源文件、16 个测试源文件；全量 88 tests，0 failures，0 errors，0 skipped；Jar 打包成功。
- target/surefire-reports/TEST-com.yizhaoqi.smartpai.benchmark.EnterpriseRagGenerationConsistencyTest.xml 中 java.version 为 17.0.20.1。
- git diff --check：通过。最终仅测试代码及本验收文档有改动。

测试报告：target/surefire-reports/。
产物：target/paismart-enterprise-rag.jar。
产物 SHA256：7ffd3f1a9e9fafcda95744424a17428c214553bb73e32a957ebafa6d6fcf8e97。
修复后测试源文件 SHA256：7997f94b79fadc871e83e0fa296989da93b0407d31465eab807e944f31554fcc。

## 环境边界

全局 ~/.m2/settings.xml 未修改。此次通过 -s 显式使用 Maven 自带的合法 settings 配置，避免已报告的全局配置格式问题；这不代表全局配置已经修复。

系统默认 Java 未修改。Homebrew 的 JDK 17 安装下载超时、没有完成；最后使用独立 Corretto 17 包完成验收。包来源为 AWS 官方 Corretto 下载地址，下载包 SHA256 已与官方校验值逐字核对：44e16fd802661560640fc04209f72a61816bae3b02aae9fa7668ee23228b22f6。

JDK 位于 /tmp，可能被系统清理。以后复跑应使用仍存在的 JDK 17 路径，不能把系统默认 JDK 23 或仅有 release=17 配置当作 JDK 17 运行验收。

全量 verify 尚有 shade 插件关于 module-info、MANIFEST、LICENSE、NOTICE 重复项的警告，本次未处理；构建仍成功。测试通过说明 Java 构建和这些回归场景通过，不代表 RAG 在线效果或真实索引的端到端指标已重新评测。
