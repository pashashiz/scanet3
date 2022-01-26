package scanet.utils

object Tabulator {

  def format(table: Seq[Seq[Any]], header: Boolean = true): String = table match {
    case Nil => ""
    case _ =>
      val sizes = table.map(row => row.map(cell => if (cell == null) 0 else cell.toString.length))
      val colSizes = for (col <- sizes.transpose) yield col.max
      val rows = for (row <- table) yield formatRow(row, colSizes)
      val separator = rowSeparator(colSizes)
      val rowsWithSeparators = separator ::
        rows.head ::
        (if (header) List(separator) else Nil) :::
        rows.tail.toList :::
        separator ::
        Nil
      rowsWithSeparators.mkString("\n")
  }

  private def formatRow(row: Seq[Any], colSizes: Seq[Int]) = {
    val cells =
      for {
        (item, size) <- row.zip(colSizes)
      } yield if (size == 0) "" else ("%-" + size + "s").format(item)
    cells.mkString("|", "|", "|")
  }

  private def rowSeparator(colSizes: Seq[Int]) =
    colSizes map { "-" * _ } mkString ("+", "+", "+")
}
