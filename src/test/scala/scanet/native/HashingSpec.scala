package scanet.native

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scanet.native.Hashing._

class HashingSpec extends AnyFlatSpec with Matchers {

  "int" should "be masked" in {
    mask(123) should be(-1552356648)
  }

  "mask" should "be recoverable" in {
    unmask(mask(123)) should be(123)
  }

  "crc32" should "compute right hash" in {
    crc32("c".getBytes()) should be(112844655)
  }

  "crc32c" should "compute right hash" in {
    crc32c("c".getBytes()) should be(552285127)
  }

  "crc32c masked" should "compute right hash" in {
    crc32cMasked(Array[Byte](24, 0, 0, 0, 0, 0, 0, 0)) should be(575373219)
  }
}
