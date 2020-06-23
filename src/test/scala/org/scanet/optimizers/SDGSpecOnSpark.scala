package org.scanet.optimizers

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{FloatType, StructField, StructType}
import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.math.syntax._
import org.scanet.models.Regression
import org.scanet.optimizers.Effect.logResult
import org.scanet.optimizers.syntax._
import org.scanet.test.CustomMatchers

class SDGSpecOnSpark extends AnyFlatSpec with CustomMatchers {

  "Adam" should "minimize linear regression" in {

    val spark = SparkSession.builder()
      .appName("example")
      .master("local[*]")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryo.registrator", "org.scanet.optimizers.TensorSerializer")
      .getOrCreate()

    val ds = spark.read
      .schema(StructType(Array(
        StructField("x", FloatType, nullable = false),
        StructField("y", FloatType, nullable = false)
      )))
      .csv(resource("linear_function_1.scv"))
      .rdd
      .map(row => Array[Float](row.getFloat(0), row.getFloat(1)))
      .cache()

    val weights = SparkOptimizer
      .minimize(Regression.linear)
      .using(SGD())
      .on(ds)
      .batch(100)
//      .each(1.epochs, logResult())
//      .each(Condition(_ => true), logResult())
      .stopAfter(50.epochs)
      .build
      .run()

    println(weights)
    val regression = Regression.linear.result.compile()
    val result = regression(BatchingIterator(ds.collect.iterator, 100, 2).next(), weights)
    result.toScalar should be <= 4.5f
  }

  private def resource(path: String): String = {
    getClass.getClassLoader.getResource(path).toString
  }
}
