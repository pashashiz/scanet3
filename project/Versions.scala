object V {
  // Spark is selected with SPARK_VERSION env variable,
  // Spark 3 runs on JDK 11+, Spark 4 requires JDK 17+
  val sparkAll: Seq[String] = Seq("3.5.10", "4.3.0")
  val spark: String = sys.env.getOrElse("SPARK_VERSION", "3.5.10")
  if (!sparkAll.contains(spark))
    sys.error(s"Unsupported SPARK_VERSION: $spark. Supported versions: ${sparkAll.mkString(", ")}")
  val sparkMajor: String = spark.split('.').head
  val java: Int = sys.props("java.version").split("\\.")(0).toInt
  if (sparkMajor == "4" && java < 17)
    sys.error(s"Spark $spark requires JDK 17+, but running on JDK $java")
  val scala = "2.13.18"
}
