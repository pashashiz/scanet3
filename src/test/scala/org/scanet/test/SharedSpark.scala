package org.scanet.test

import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, Suite}

trait SharedSpark extends BeforeAndAfterAll {

  self: Suite =>

  @transient private var _spark: SparkSession = _

  def spark: SparkSession = _spark
  def sc: SparkContext = _spark.sparkContext

  protected implicit def reuseContextIfPossible: Boolean = false

  def appID: String = (this.getClass.getName
    + math.floor(math.random * 10E4).toLong.toString)

  def conf = {
    new SparkConf()
      .setMaster("local[*]")
      .setAppName("test")
      .set("spark.ui.enabled", "false")
      .set("spark.app.id", appID)
      .set("spark.driver.host", "localhost")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "org.scanet.optimizers.TensorSerializer")
  }

  override def beforeAll() {
    _spark = SparkSession.builder().config(conf).getOrCreate()
    super.beforeAll()
  }

  override def afterAll() {
    try {
      if (!reuseContextIfPossible) {
        stop(_spark)
        _spark = null
      }
    } finally {
      super.afterAll()
    }
  }

  def stop(sc: SparkSession) {
    Option(sc).foreach {ctx => ctx.stop()}
    System.clearProperty("spark.driver.port")
  }
}
