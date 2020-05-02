name := "scanet3"

version := "0.1"

scalaVersion := "2.13.2"

//scalacOptions += "-Ypartial-unification"
scalacOptions += "-Ymacro-annotations"

//addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

libraryDependencies ++= Seq(
  "org.typelevel" %% "spire" % "0.17.0-M1",
  "org.typelevel" %% "cats-core" % "2.0.0",
  "org.typelevel" %% "simulacrum" % "1.0.0",
  "org.tensorflow" % "tensorflow" % "1.15.0",
  "org.tensorflow" % "proto" % "1.15.0",
  "com.google.guava" % "guava" % "29.0-jre", // needed for crc32c only
  "org.scalacheck" %% "scalacheck" % "1.14.3" % "test",
  "org.scalatest" %% "scalatest" % "3.1.1" % "test"
)

scalacOptions ++= Seq(
  "-Xlint",
  "-deprecation",
  "-Ywarn-unused:params,-implicits",
  "-feature",
  "-language:implicitConversions"
)