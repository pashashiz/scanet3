package org.scanet.optimizers

import org.scalatest.wordspec.AnyWordSpec
import org.scanet.core.Tensor
import org.scanet.core.syntax._
import org.scanet.test.KryoMatchers

class KryoSerializersSpec extends AnyWordSpec with KryoMatchers {

  val kryo = KryoSerializers.Kryo

  "kryo" should {

    "write/read the tensor of floats" in {
      Tensor.vector(1f, 2f, 3f) should beSerializableAs[Tensor[Float]](using = kryo)
    }

    "write/read the tensor of strings" in {
      Tensor.vector("hey", "dawg") should beSerializableAs[Tensor[String]](using = kryo)
    }
  }
}
