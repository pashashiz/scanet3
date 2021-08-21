package org.scanet.test

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import org.scalatest.matchers.should.Matchers
import org.scalatest.matchers.{MatchResult, Matcher}

import java.io.ByteArrayOutputStream
import scala.reflect.{ClassTag, classTag}
import scala.util.Try

trait KryoMatchers extends Matchers {

  def beSerializableAs[T: ClassTag](using: Kryo): Matcher[T] = {

    def serialize(obj: T): Either[String, Array[Byte]] =
      Try {
        val bytesOutput = new ByteArrayOutputStream()
        val kryoOutput = new Output(bytesOutput)
        using.writeObject(kryoOutput, obj)
        kryoOutput.flush()
        kryoOutput.close()
        bytesOutput.toByteArray
      }.toEither.left.map { e =>
        s"The object $obj is not serializable. ${e.getMessage}. "
      }

    def deserialize(bytes: Array[Byte]): Either[String, T] =
      Try {
        val kryoInput = new Input(bytes)
        val clazz = classTag[T].runtimeClass.asInstanceOf[Class[T]]
        using.readObject[T](kryoInput, clazz)
      }.toEither.left.map { e =>
        s"Cannot deserialize the object. ${e.getMessage}"
      }

    def checkEquals(original: T, deserialized: T): Either[String, Unit] =
      Either.cond(
        deserialized == original,
        (),
        s"The deserialized object $deserialized is not equal to the original $original")

    obj => {
      val verified = for {
        bytes <- serialize(obj)
        deserialized <- deserialize(bytes)
        _ <- checkEquals(obj, deserialized)
      } yield ()
      val message = verified.fold(identity, _ => s"The object $obj is serializable")
      MatchResult(verified.isRight, message, message)
    }
  }
}
