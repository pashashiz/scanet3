package scanet.utils

import org.scalatest.matchers.should.Matchers
import org.scalatest.wordspec.AnyWordSpec

class TabulatorSpec extends AnyWordSpec with Matchers {

  "tabulator" should {
    "draw a table" in {
      val table = Seq(
        Seq("name", "age"),
        Seq("Pavlo", "31"),
        Seq("Yurii Dawg", "28"))
      Tabulator.format(table) shouldBe
      """#+----------+---+
         #|name      |age|
         #+----------+---+
         #|Pavlo     |31 |
         #|Yurii Dawg|28 |
         #+----------+---+"""
        .stripMargin('#')
    }
  }
}
