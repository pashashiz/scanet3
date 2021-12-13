package scanet.math

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scanet.core.Convertible
import scanet.math.syntax._

class ConvertibleSpec extends AnyFlatSpec with Matchers {

  "Float" should "be converted to Int" in {
    Convertible[Float, Int].convert(2.5f) should be(2)
  }
}
