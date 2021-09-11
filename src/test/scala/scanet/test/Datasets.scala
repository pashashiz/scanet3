package scanet.test

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{FloatType, StructField, StructType}
import scanet.{core, datasets}

trait Datasets {

  self: SharedSpark =>

  def zero: RDD[Array[Float]] = spark.sparkContext.parallelize(Seq[Array[Float]]())

  lazy val linearFunction: RDD[Array[Float]] = {
    spark.read
      .schema(
        StructType(
          Array(
            StructField("x", FloatType, nullable = false),
            StructField("y", FloatType, nullable = false))))
      .csv(resource("linear_function_1.scv"))
      .rdd
      .map(row => Array[Float](row.getFloat(0), row.getFloat(1)))
      .coalesce(1)
      .cache()
  }

  lazy val logisticRegression: RDD[Array[Float]] = {
    spark.read
      .schema(
        StructType(
          Array(
            StructField("x1", FloatType, nullable = false),
            StructField("x2", FloatType, nullable = false),
            StructField("y", FloatType, nullable = false))))
      .csv(resource("logistic_regression_1.scv"))
      .rdd
      .map(row => Array[Float](row.getFloat(0), row.getFloat(1), row.getFloat(2)))
      .coalesce(1)
      .cache()
  }

  lazy val facebookComments: RDD[Array[Float]] =
    spark.read
      .csv(resource("facebook-comments-scaled.csv"))
      .rdd
      .map(row => row.toSeq.map(v => v.asInstanceOf[String].toFloat).toArray)
      .coalesce(1)
      .cache()

  val MNISTMemoized = core.memoize((trainingSize: Int, testSize: Int) => {
    val (training, test) = datasets.MNIST.load(sc, trainingSize, testSize)
    (training.coalesce(1).cache(), test.coalesce(1).cache())
  })

  def MNIST(trainingSize: Int = 60000, testSize: Int = 10000) =
    MNISTMemoized(trainingSize, testSize)

  def resource(path: String): String = {
    getClass.getClassLoader.getResource(path).toString
  }
}
