package scanet.math

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scanet.math.Generator.uniform
import scanet.math.syntax._

class RandomSpec extends AnyFlatSpec with Matchers {

  "uniform distribution" should "work for Int with one next" in {
    val r1 = Random[Int](uniform(1L))
    val (r2, v1) = r1.next
    val (_, v2) = r2.next
    v1 should be(384748)
    v2 should be(-1151252339)
  }

  it should "work for Int with multiple next" in {
    val (_, values) = Random[Int](uniform(1L)).next(3)
    values should be(Array(384748, -1151252339, -549383847))
  }

  it should "work for Float with default [0, 1] range" in {
    val (_, values) = Random[Float](uniform(1L)).next(3)
    values should be(Array(8.952618e-5f, 0.73195314f, 0.8720866f))
  }

  it should "work for Float with custom [-1, 1] range" in {
    val (_, values) = Random[Float](uniform(1L), range = Some((-1f, 1f))).next(3)
    values should be(Array(-0.99982095f, 0.4639063f, 0.74417317f))
  }
}
