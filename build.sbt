name := "scanet3"
version := "0.1"
scalaVersion := "2.12.14"

resolvers ++= Seq("sonatype-snapshots" at "https://oss.sonatype.org/content/repositories/snapshots")

addCommandAlias("testFast", "testOnly -- -l org.scalatest.tags.Slow")

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.1" cross CrossVersion.full)
// scalacOptions += "-Ymacro-annotations" // NOTE: uncomment with scala 2.13 again

libraryDependencies ++= Seq(
  "org.scala-lang" % "scala-reflect" % "2.12.14", // NOTE: remove with scala 2.13
  "org.typelevel" %% "simulacrum" % "1.0.0",
  "org.tensorflow" % "tensorflow-core-platform" % "0.5.0-SNAPSHOT",
  "com.google.guava" % "guava" % "29.0-jre", // NOTE: needed for crc32c only, need to reimplement and remove the dependency
  "org.apache.spark" %% "spark-core" % "3.1.2",
  "org.apache.spark" %% "spark-sql" % "3.1.2",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.4",
  "ch.qos.logback" % "logback-classic" % "1.2.10",
  "org.scalacheck" %% "scalacheck" % "1.15.4" % "test",
  "org.scalatest" %% "scalatest" % "3.2.9" % "test"
)

// temp workaround built locally for macosx-arm64, until the official one is published
Compile / unmanagedJars += file("tensorflow-core-api-0.5.0-SNAPSHOT-macosx-arm64.jar")

updateOptions := updateOptions.value.withLatestSnapshots(false)

scalacOptions ++= Seq(
  "-Xlint",
  "-deprecation",
  "-Ypartial-unification",
  "-Ywarn-unused:params,-implicits",
  "-feature",
  "-language:implicitConversions",
  "-language:higherKinds",
  "-language:existentials"
)

Test / parallelExecution := false
Test / testOptions ++= Seq(
  Tests.Argument(TestFrameworks.ScalaTest, "-oD"),
  Tests.Argument(TestFrameworks.ScalaTest, "-oF")
)

Test / fork := true
javaOptions ++= Seq("-Xms512M", "-Xmx2048M", "-XX:+CMSClassUnloadingEnabled")
