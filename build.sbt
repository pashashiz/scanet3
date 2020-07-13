name := "scanet3"

version := "0.1"

scalaVersion := "2.12.11"

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.1" cross CrossVersion.full)
// scalacOptions += "-Ymacro-annotations" // NOTE: uncomment with scala 2.13 again

libraryDependencies ++= Seq(
  // "org.typelevel" %% "spire" % "0.14.1", // NOTE: use 0.17.0-M1 with scala 2.13 again
  //"org.typelevel" %% "cats-core" % "2.0.0",
  "org.scala-lang" % "scala-reflect" % "2.12.11", // NOTE: remove with scala 2.13
  "org.typelevel" %% "simulacrum" % "1.0.0",
  "org.tensorflow" % "tensorflow" % "1.15.0",
  "org.tensorflow" % "proto" % "1.15.0",
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

parallelExecution in Test := false

fork in Test := true
javaOptions ++= Seq("-Xms512M", "-Xmx2048M", "-XX:+CMSClassUnloadingEnabled")
