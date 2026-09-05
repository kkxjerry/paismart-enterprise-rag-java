# Source-only build artifact. Execute the export on A40, never on the Mac.
# No base image download, processes, services, or model credentials are needed.
# BuildKit exports this audited develop snapshot to a NEW server build directory.
# Compile and run Maven with the server's JDK 17 after the export.
FROM scratch AS source
COPY pom.xml /pom.xml
COPY src/ /src/
# Regression tests load tracked index mappings and experiment configs from disk.
COPY config/ /config/
COPY docs/A40_BUILD_SOURCE_MANIFEST.json /SOURCE_MANIFEST.json
COPY docs/GENERATION_CONSISTENCY_TEST_FIX_2026-09-04.md /docs/GENERATION_CONSISTENCY_TEST_FIX_2026-09-04.md
COPY docs/A40_GENERATION_CONSISTENCY_VALIDATION_2026-09-04.md /docs/A40_GENERATION_CONSISTENCY_VALIDATION_2026-09-04.md
