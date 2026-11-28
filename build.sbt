name := "scanet3"
version := "0.1.0-SNAPSHOT"

scalaVersion := V.scala

resolvers ++= Seq("sonatype-snapshots" at "https://oss.sonatype.org/content/repositories/snapshots")

addCommandAlias("testFast", "testOnly -- -l org.scalatest.tags.Slow")

libraryDependencies ++= Seq(
  "org.typelevel" %% "simulacrum" % "1.0.1",
  "org.tensorflow" % "tensorflow-core-platform" % "1.2.0",
  // NOTE: needed for crc32c only, need to reimplement and remove the dependency
  "com.google.guava" % "guava" % "33.4.8-jre",
  // logback is the logging backend, so Spark log4j to SLF4J bridge is excluded
  ("org.apache.spark" %% "spark-sql" % V.spark).exclude("org.apache.logging.log4j", "log4j-slf4j2-impl"),
  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.5",
  "ch.qos.logback" % "logback-classic" % "1.5.18",
  "com.github.ben-manes.caffeine" % "caffeine" % "2.8.5",
  "org.scalacheck" %% "scalacheck" % "1.17.0" % Test,
  "org.scalatest" %% "scalatest" % "3.2.19" % Test)

updateOptions := updateOptions.value.withLatestSnapshots(false)

Compile / scalacOptions ++= Seq(
  "-Xlint",
  "-deprecation",
  "-Ywarn-unused:explicits",
  "-Wconf:cat=unused-nowarn:s",
  "-feature",
  "-language:implicitConversions",
  "-language:higherKinds",
  "-language:existentials",
  "-Ymacro-annotations")

// code which differs between Spark versions, if any
Compile / unmanagedSourceDirectories += baseDirectory.value / "src/main" / s"scala-spark${V.sparkMajor}"

Test / parallelExecution := false
Test / testOptions ++= Seq(
  Tests.Argument(TestFrameworks.ScalaTest, "-oD"),
  Tests.Argument(TestFrameworks.ScalaTest, "-oF"))

Test / fork := true
Test / javaOptions ++= Seq(
  "-Xms512M",
  "-Xmx2048M",
  "-Dcom.sun.management.jmxremote",
  "-Dcom.sun.management.jmxremote.port=9010",
  "-Dcom.sun.management.jmxremote.host=localhost",
  "-Djava.rmi.server.hostname=localhost",
  "-Dcom.sun.management.jmxremote.rmi.port=9010",
  "-Dcom.sun.management.jmxremote.authenticate=false",
  "-Dcom.sun.management.jmxremote.ssl=false")

// Spark on JDK 17+ needs access to JDK internals,
// borrowed from https://github.com/apache/spark/blob/master/project/SparkBuild.scala
Test / javaOptions ++= {
  if (V.java >= 17)
    Seq(
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
      "--add-opens=java.base/java.io=ALL-UNNAMED",
      "--add-opens=java.base/java.net=ALL-UNNAMED",
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/java.util=ALL-UNNAMED",
      "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
      "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
      "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
      "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED")
  else Seq.empty
}