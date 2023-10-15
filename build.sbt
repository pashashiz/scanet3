name := "scanet3"
version := "0.1.0-SNAPSHOT"

scalaVersion := "2.13.10"
crossScalaVersions := Seq("2.12.17", "2.13.10")

resolvers ++= Seq("sonatype-snapshots" at "https://oss.sonatype.org/content/repositories/snapshots")

addCommandAlias("testFast", "testOnly -- -l org.scalatest.tags.Slow")

libraryDependencies ++= Seq(
  "org.typelevel" %% "simulacrum" % "1.0.1",
  "org.tensorflow" % "tensorflow-core-platform" % "0.5.0",
  // NOTE: needed for crc32c only, need to reimplement and remove the dependency
  "com.google.guava" % "guava" % "29.0-jre",
  "org.apache.spark" %% "spark-sql" % "3.3.1",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.5",
  "ch.qos.logback" % "logback-classic" % "1.4.5",
  "org.scala-lang.modules" %% "scala-collection-compat" % "2.9.0",
  "com.github.ben-manes.caffeine" % "caffeine" % "2.8.5",
  "org.scalacheck" %% "scalacheck" % "1.17.0" % Test,
  "org.scalatest" %% "scalatest" % "3.2.14" % Test)

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

// temp workaround built locally for macosx-arm64, until the official one is published
Compile / unmanagedJars += file("tensorflow-core-api-0.5.0-SNAPSHOT-macosx-arm64.jar")

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
  "-XX:+CMSClassUnloadingEnabled",
  "-Dcom.sun.management.jmxremote",
  "-Dcom.sun.management.jmxremote.port=9010",
  "-Dcom.sun.management.jmxremote.host=localhost",
  "-Djava.rmi.server.hostname=localhost",
  "-Dcom.sun.management.jmxremote.rmi.port=9010",
  "-Dcom.sun.management.jmxremote.authenticate=false",
  "-Dcom.sun.management.jmxremote.ssl=false")