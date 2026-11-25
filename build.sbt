name := "scanet3"
version := "0.1.0-SNAPSHOT"

scalaVersion := "2.13.16"
crossScalaVersions := Seq("2.12.17", "2.13.16")

resolvers ++= Seq("sonatype-snapshots" at "https://oss.sonatype.org/content/repositories/snapshots")

addCommandAlias("testFast", "testOnly -- -l org.scalatest.tags.Slow")

libraryDependencies ++= Seq(
  "org.typelevel" %% "simulacrum" % "1.0.1",
  "org.tensorflow" % "tensorflow-core-platform" % "1.2.0",
  // NOTE: needed for crc32c only, need to reimplement and remove the dependency
  "com.google.guava" % "guava" % "33.4.8-jre",
  "org.apache.spark" %% "spark-sql" % "3.5.10",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.5",
  "ch.qos.logback" % "logback-classic" % "1.5.18",
  "org.scala-lang.modules" %% "scala-collection-compat" % "2.13.0",
  "com.github.ben-manes.caffeine" % "caffeine" % "2.8.5",
  "org.scalacheck" %% "scalacheck" % "1.17.0" % Test,
  "org.scalatest" %% "scalatest" % "3.2.19" % Test)

libraryDependencies ++= {
  CrossVersion.partialVersion(scalaVersion.value) match {
    case Some((2, n)) if n >= 13 =>
      Nil
    case _ =>
      Seq(
        "org.scala-lang" % "scala-reflect" % scalaVersion.value % Provided,
        compilerPlugin("org.scalamacros" % "paradise" % "2.1.1" cross CrossVersion.full))
  }
}

updateOptions := updateOptions.value.withLatestSnapshots(false)

Compile / scalacOptions ++= Seq(
  "-Xlint",
  "-deprecation",
  "-Ywarn-unused:explicits",
  "-Wconf:cat=unused-nowarn:s",
  "-feature",
  "-language:implicitConversions",
  "-language:higherKinds",
  "-language:existentials")

Compile / scalacOptions ++= {
  CrossVersion.partialVersion(scalaVersion.value) match {
    case Some((2, n)) if n >= 13 => Seq("-Ymacro-annotations")
    case _                       => Seq("-Ypartial-unification")
  }
}

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