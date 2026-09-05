# A40 Generation 测试修复验收

最终全量构建完成：2026-09-05 05:15:37 UTC / 洛杉矶 2026-09-04 22:15:37。

## 结论

本次在 A40 主机 a40-48g-3 上使用原生 Ubuntu OpenJDK 17.0.19、Maven 3.6.3 完成验收，不是引用 Mac 的测试结果。test-compile 通过；Generation 指定类 14/14 通过；clean verify 全量 88/88 通过，0 failures、0 errors、0 skipped，Jar 打包完成。

没有修改生产 Java 源码或 pom.xml，没有恢复废弃接口，没有削弱或跳过测试。没有调用云生成 API、真实 embedding 或修改 Elasticsearch 索引。本次没有发布新服务，也没有重启现有服务。

## 服务器源码与产物

新构建目录：

/srv/paismart-develop-experiment/repro-adaptive-v1/builds/209fc00-generation-test-fix

这里是 Mac develop 的可核验 Java 构建快照：基线 209fc0046caeb706c8dea9b29f3844d15b1a8bd5，加 EnterpriseRagGenerationConsistencyTest.java 的未提交修复。不是一个新的 Git commit，也不是 Git checkout；本轮未提交或推送。包括 pom.xml、全部 34 个 Java 源文件和全部 6 个已跟踪 config 文件。Python 工具和评测数据未在此复制。

SOURCE_MANIFEST.json 保存基线提交、修复文件哈希、补丁哈希和源码/config 整体校验值。Mac 与 A40 的源码和配置哈希一致。

源码树 SHA256（pom.xml + src，校验算法见 manifest）：
5543bcfe9395fd39fc18cf1d34a2850ea3bfc34e783f50d9a4bb5b5c46d6cc96

配置树 SHA256：
79cdc2924d5eb08c02668cc17bd935820adba3d620064fe4d94a3ee875429865

新 Jar：

target/paismart-enterprise-rag.jar

Jar SHA256：
3559bbf48c50cbfc042ac011ef29a26da5189dcbff392b623c5f5bbb8381f94e

Jar MANIFEST 的 Build-Jdk 为 17.0.19；Surefire XML 的 java.version 也是 17.0.19。

源码包通过 A40 上的 BuildKit source-only 构建导出到独立目录；没有在 Mac 运行 Docker，没有下载基础镜像，没有启动应用容器。Maven 编译及测试直接运行在 A40 主机，而不是容器内。

## 原生验收命令与日志

```sh
cd /srv/paismart-develop-experiment/repro-adaptive-v1/builds/209fc00-generation-test-fix
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
SETTINGS=/srv/paismart-develop-experiment/maven-settings.xml
mvn -v
mvn -B -ntp -s "$SETTINGS" -l build-test-compile-mirror.log -DskipTests test-compile
mvn -B -ntp -s "$SETTINGS" -l build-generation-test.log -Dmaven.test.redirectTestOutputToFile=true -Dtest=EnterpriseRagGenerationConsistencyTest test
mvn -B -ntp -s "$SETTINGS" -l build-clean-verify-complete.log -Dmaven.test.redirectTestOutputToFile=true clean verify
```

最终全量日志：build-clean-verify-complete.log。
测试明细：target/surefire-reports/，16 个测试类，88 个执行用例。

全量编译重新编译 18 个生产源文件、16 个测试源文件。首次成功指定类测试耗时 6.338 秒；最终全量构建耗时 10.740 秒。仍有 shade 的 module-info、MANIFEST、LICENSE、NOTICE 重复项警告，未将其描述为已消除。

## 本轮真实阻塞与处理

第一次使用 /etc/maven/settings.xml 时，Maven Central 的插件请求失败：Remote host terminated the handshake；curl 直连同一 HTTPS 地址也出现 SSL connection timeout。此时尚未进入 Java 编译。改为显式使用服务器早已存在的 /srv/paismart-develop-experiment/maven-settings.xml（aliyun-public 镜像）后构建成功。没有关闭 TLS 校验，没有修改全局 settings，也没有断言网络根因已经修复。原始失败日志：build-test-compile.log。

第一次全量测试失败于两个配置文件读取场景：source-only 同步包漏带 config 目录。这是本轮打包遗漏，不是原生产代码的新缺陷。补齐全部已跟踪 config 文件后重新 clean verify，88 个测试全部通过。失败日志 build-clean-verify.log 保留，最终通过日志另存 build-clean-verify-complete.log；没有删除失败证据。

## 现有 A40 环境实时核对

查到的可追溯旧源码仓库：
/srv/paismart-develop-experiment/repro-p0p1/full500/source
其 develop 为 035a7d9，工作区干净；没有 GenerationConsistency 测试文件。本轮没有修改它。

repro-adaptive-v1/bin 的四个历史 Jar 在 MANIFEST 中均未记录 Git commit，Build-Jdk-Spec 为 23。构建 JDK 不等于字节码 release 版本，不能因此认定它们不能在 JDK 17 运行；也不能把 latest/final/release-candidate 文件名当作提交版本证明。

18090 实时监听 PID 2462717，使用 bin/paismart-enterprise-rag.jar；索引为 paismart_adaptive_lifecycle_test_v1，embedding 为 http://127.0.0.1:18086/v1/embeddings、模型 fake-embedding。/health 返回 status=ok 和该测试索引。这仍然是测试服务，不是正式 RAG 服务。

操作前后原有 19200、18083、18084、18085、18086、18090 监听 PID 均未变化。bin 下四个历史 Jar 的 SHA256 操作前后完全相同：

- paismart-enterprise-rag.jar：91746405e41337ea760d7004a86babf2f0355899ce7786c99f5cec5817cff7ae
- paismart-enterprise-rag-latest.jar：2457e19f2358581842080c69299ad9d89aa893e80d4bcb00802d190c4fcdd52e
- paismart-enterprise-rag-final.jar：8ad73fb74eb53fa9695ade0d5d3264288ee2127632a9b4a1c2a99d1f3de7e853
- paismart-enterprise-rag-release-candidate.jar：70c644d7509f2013ae05ab35274749d6b0c4a0406932dcd42a57065da4326c70

验收范围仅为当前修复版的 A40 Java 构建、回归测试和打包。没有重跑真实 RAG 问答质量评测，没有把 18090 改为正式索引或真实 embedding，没有替换在用 Jar。
