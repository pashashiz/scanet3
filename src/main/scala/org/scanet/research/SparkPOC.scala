package org.scanet.research

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{FloatType, StructField, StructType}
import org.scanet.core.Tensor
import org.scanet.math.syntax._
import org.scanet.optimizers.BatchingIterator

object SparkPOC {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("example")
      .master("local[*]")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryo.registrator", "org.scanet.optimizers.TensorSerializer")
      .getOrCreate()
    val sc = spark.sparkContext

    val rdd = spark.read
        .schema(StructType(Array(
          StructField("x", FloatType, nullable = false),
          StructField("y", FloatType, nullable = false)
        )))
      .csv(resource("linear_function_1.scv"))
      .rdd
      .map(row => Array[Float](row.getFloat(0), row.getFloat(1)))
      .cache()

    val epochs = 10
    val batch = 25
    val columns = 2
    var weights = sc.broadcast(Tensor.zeros[Float](batch, columns))
    (0 until epochs).foreach(epoch => {
      println(s"#$epoch")
      val newWeights = rdd
        .repartition(1)
        .mapPartitions(it => {
          val batches = BatchingIterator(it, batch, columns)
          val emulatedWeights = batches.foldLeft(weights.value)(
            (l, r) => ((l.const + r.const) / 2.0f.const).eval)
          println("Thread: " + Thread.currentThread().getName)
          Iterator(emulatedWeights)
        })
        .treeReduce((left, right) => ((left.const + right.const) / 2.0f.const).eval)
      weights = sc.broadcast(newWeights)
      println(newWeights)
    })
  }

  private def resource(path: String): String = {
    SparkPOC.getClass.getClassLoader.getResource(path).toString
  }
}
