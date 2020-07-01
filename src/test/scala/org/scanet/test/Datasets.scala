package org.scanet.test

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{FloatType, StructField, StructType}

trait Datasets {

  self: SharedSpark =>

  def zero: RDD[Array[Float]] = spark.sparkContext.parallelize(Seq[Array[Float]]())

  def linearFunction: RDD[Array[Float]] = {
    spark.read
      .schema(StructType(Array(
        StructField("x", FloatType, nullable = false),
        StructField("y", FloatType, nullable = false)
      )))
      .csv(resource("linear_function_1.scv"))
      .rdd
      .map(row => Array[Float](row.getFloat(0), row.getFloat(1)))
  }

  def logisticRegression: RDD[Array[Float]] = {
    spark.read
      .schema(StructType(Array(
        StructField("x1", FloatType, nullable = false),
        StructField("x2", FloatType, nullable = false),
        StructField("y", FloatType, nullable = false)
      )))
      .csv(resource("logistic_regression_1.scv"))
      .rdd
      .map(row => Array[Float](row.getFloat(0), row.getFloat(1), row.getFloat(2)))
  }

  def facebookComments: RDD[Array[Float]] = {
    spark.read
      .csv(resource("facebook-comments-scaled.csv"))
      .rdd
      .map(row => row.toSeq.map(v => v.asInstanceOf[String].toFloat).toArray)
  }

  private def resource(path: String): String = {
    getClass.getClassLoader.getResource(path).toString
  }
}
