package scanet.test

import org.jfree.data.general.DefaultPieDataset
import org.jfree.chart._
import org.jfree.chart.plot.{DrawingSupplier, PiePlot}
import scanet.test.Colors._

import java.awt.{Color, Font, GraphicsEnvironment, Paint, Shape, Stroke}
import java.io.File

object Colors {

  val Red = new Color(0xf44336)
  val Pink = new Color(0xe91e63)
  val Purple = new Color(0x9c27b0)
  val DeepPurple = new Color(0x673ab7)
  val Indigo = new Color(0x3f51b5)
  val Blue = new Color(0x2196f3)
  val LightBlue = new Color(0x03a9f4)
  val Cyan = new Color(0x00bcd4)
  val Teal = new Color(0x009688)
  val Green = new Color(0x4caf50)
  val LightGreen = new Color(0x8bc34a)
  val Lime = new Color(0xcddc39)
  val Yellow = new Color(0xffeb3b)
  val Amber = new Color(0xffc107)
  val Orange = new Color(0xff9800)
  val DeepOrange = new Color(0xff5722)

  val All = Seq(
    Red,
    Pink,
    Purple,
    DeepPurple,
    Indigo,
    Blue,
    LightBlue,
    Cyan,
    Teal,
    Green,
    LightGreen,
    Lime,
    Yellow,
    Amber,
    Orange,
    DeepOrange)

}

class MaterialDrawingSupplier extends DrawingSupplier {

  private def infinite(colors: Seq[Color], fallback: Seq[Color]): Stream[Color] = {
    colors match {
      case head :: Nil  => head #:: infinite(fallback.tail, fallback)
      case head :: tail => head #:: infinite(tail, fallback)
      case Nil          => Stream.Empty
    }

  }
  private val outlierColors = infinite(Colors.All.reverse, Colors.All.reverse).iterator

  override def getNextPaint: Paint = outlierColors.next()

  override def getNextOutlinePaint: Paint = ???
  override def getNextFillPaint: Paint = ???
  override def getNextStroke: Stroke = ???
  override def getNextOutlineStroke: Stroke = ???
  override def getNextShape: Shape = ???
}

object Charts {
  def main(args: Array[String]): Unit = {
    val dataset = new DefaultPieDataset[String]()
    dataset.setValue("Python", 11.77)
    dataset.setValue("C", 11.72)
    dataset.setValue("Java", 11.72)
    dataset.setValue("C++", 8.28)
    dataset.setValue("C#", 6.06)
    dataset.setValue("JS", 2.66)
    dataset.setValue("PHP", 1.81)
    dataset.setValue("Ruby", 1.43)
    dataset.setValue("Swift", 1.43)
    dataset.setValue("Scala", 0.36)
    val chart = ChartFactory.createPieChart("TIOBE Index", dataset, true, true, false)
    val plot = chart.getPlot.asInstanceOf[PiePlot[String]]
    plot.setBackgroundPaint(Color.WHITE)
    plot.setOutlineVisible(false)
    plot.setShadowPaint(null)
    plot.setLabelBackgroundPaint(Color.WHITE)
    plot.setLabelOutlineStroke(null)
    plot.setLabelShadowPaint(null)
    plot.setLabelFont(new Font("Arial", Font.PLAIN, 18))
    plot.setDrawingSupplier(new MaterialDrawingSupplier)
    ChartUtils.saveChartAsJPEG(new File("Pie_Chart.jpeg"), chart, 1000, 800)
  }
}
