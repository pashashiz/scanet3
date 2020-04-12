name := "scanet3"

version := "0.1"

scalaVersion := "2.12.8"

//scalacOptions += "-Ypartial-unification"

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

libraryDependencies ++= Seq(
  "org.typelevel" %% "spire" % "0.14.1",
  "org.typelevel" %% "cats-core" % "2.0.0",
  "org.typelevel" %% "simulacrum" % "1.0.0",
  "org.tensorflow" % "tensorflow" % "1.15.0",
  "org.scalacheck" %% "scalacheck" % "1.14.3" % "test",
  "org.scalatest" %% "scalatest" % "3.1.1" % "test"
)

scalacOptions ++= Seq(
  "-Xlint",
  "-deprecation",
  "-feature",
  "-language:implicitConversions"
)