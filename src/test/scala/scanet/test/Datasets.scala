package scanet.test

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.{FloatType, StructField, StructType}
import scanet.core.Shape
import scanet.optimizers.Record
import scanet.optimizers.syntax._
import scanet.core.syntax._

import scanet.{core, datasets}

trait Datasets {

  self: SharedSpark =>

  // NOTE: we actually need one element in a dataset, it will represent one computation in an epoch
  // and the result of such dataset will be simply ignored
  def zero: Dataset[Record[Float]] = {
    import implicits._
    spark
      .createDataset(Seq[Record[Float]](Record(Array[Float](0.0f), Array[Float](0.0f))))
      .withShapes(Shape(), Shape())
  }

  lazy val linearFunction: Dataset[Record[Float]] = {
    import implicits._
    spark.read
      .schema(
        StructType(
          Array(
            StructField("x", FloatType, nullable = false),
            StructField("y", FloatType, nullable = false))))
      .csv(resource("linear_function_1.scv"))
      .map(row => Record[Float](Array(row.getFloat(0)), Array(row.getFloat(1))))
      .withShapes(Shape(1), Shape(1))
      .coalesce(1)
      .cache()
  }

  lazy val logisticRegression: Dataset[Record[Float]] = {
    import implicits._
    spark.read
      .schema(
        StructType(
          Array(
            StructField("x1", FloatType, nullable = false),
            StructField("x2", FloatType, nullable = false),
            StructField("y", FloatType, nullable = false))))
      .csv(resource("logistic_regression_1.scv"))
      .map(row =>
        Record[Float](Array(row.getFloat(0) / 100, row.getFloat(1) / 100), Array(row.getFloat(2))))
      .withShapes(Shape(2), Shape(1))
      .coalesce(1)
      .cache()
  }

  lazy val facebookComments: Dataset[Record[Float]] = {
    import implicits._
    spark.read
      .csv(resource("facebook-comments-scaled.csv"))
      .map { row =>
        val all = row.toSeq.map(v => v.asInstanceOf[String].toFloat).toArray
        Record(all.slice(0, all.length - 1), all.slice(all.length - 1, all.length))
      }
      .withShapes(Shape(53), Shape(1))
      .coalesce(1)
      .cache()
  }

  val MNISTMemoized = core.memoize((trainingSize: Int, testSize: Int) => {
    val (training, test) = datasets.MNIST.load(trainingSize, testSize)(spark)
    (training.coalesce(1).cache(), test.coalesce(1).cache())
  })

  def MNIST(trainingSize: Int = 60000, testSize: Int = 10000) =
    MNISTMemoized(trainingSize, testSize)

  def resource(path: String): String = {
    getClass.getClassLoader.getResource(path).toString
  }
}
