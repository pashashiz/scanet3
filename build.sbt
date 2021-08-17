name := "scanet3"
version := "0.1"
scalaVersion := "2.12.12"

addCommandAlias("testFast", "testOnly -- -l org.scalatest.tags.Slow")

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.1" cross CrossVersion.full)
// scalacOptions += "-Ymacro-annotations" // NOTE: uncomment with scala 2.13 again

libraryDependencies ++= Seq(
  "org.scala-lang" % "scala-reflect" % "2.12.11", // NOTE: remove with scala 2.13
  "org.typelevel" %% "simulacrum" % "1.0.0",
  "org.tensorflow" % "tensorflow-core-platform" % "0.3.2",
  "com.google.guava" % "guava" % "29.0-jre", // NOTE: needed for crc32c only, need to reimplement and remove the dependency
  "org.apache.spark" %% "spark-core" % "3.0.0",
  "org.apache.spark" %% "spark-sql" % "3.0.0",
  "org.scalacheck" %% "scalacheck" % "1.14.3" % "test",
  "org.scalatest" %% "scalatest" % "3.1.1" % "test"
)

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
