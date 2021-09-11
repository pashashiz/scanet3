package scanet.optimizers

import org.scalatest.wordspec.AnyWordSpec
import scanet.core.Tensor
import scanet.core.syntax._
import scanet.test.KryoMatchers

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
