# Build

The project uses Scala 2.13 and supports 2 Spark versions, selected with the `SPARK_VERSION` env variable
(see `project/Versions.scala`):

| Spark            | JDK  |
|------------------|------|
| 3.5.10 (default) | 11+  |
| 4.3.0            | 17+  |

```sh
# Spark 3.5
sbt test
# Spark 4
SPARK_VERSION=4.3.0 sbt test
```

Run tests without the slow ones (model training):

```sh
sbt testFast
```

If some code has to differ between Spark versions, put it into `src/main/scala-spark3`
and `src/main/scala-spark4`, only the folder of the selected Spark version is compiled.

TensorFlow Java ships native binaries for Linux (x86_64, arm64), macOS (arm64) and Windows (x86_64).
