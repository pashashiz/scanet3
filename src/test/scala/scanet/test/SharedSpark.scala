package scanet.test

import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, Suite}

trait SharedSpark extends BeforeAndAfterAll {

  self: Suite =>

  @transient private var _spark: SparkSession = _

  def spark: SparkSession = _spark
  lazy val implicits = spark.implicits
  def sc: SparkContext = _spark.sparkContext

  implicit protected def reuseContextIfPossible: Boolean = false

  def appID: String = (this.getClass.getName
    + math.floor(math.random() * 10e4).toLong.toString)

  def conf: SparkConf = {
    new SparkConf()
      .setMaster("local[*]")
      .setAppName("test")
      .set("spark.ui.enabled", "false")
      .set("spark.ui.showConsoleProgress", "false")
      .set("spark.app.id", appID)
      .set("spark.driver.host", "localhost")
      .set("spark.sql.shuffle.partitions", "1")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "scanet.optimizers.KryoSerializers")
  }

  override def beforeAll(): Unit = {
    _spark = SparkSession.builder().config(conf).getOrCreate()
    super.beforeAll()
  }

  override def afterAll(): Unit = {
    try {
      if (!reuseContextIfPossible) {
        stop(_spark)
        _spark = null
      }
    } finally {
      super.afterAll()
    }
  }

  def stop(sc: SparkSession): Unit = {
    Option(sc).foreach { ctx => ctx.stop() }
    System.clearProperty("spark.driver.port")
  }
}
