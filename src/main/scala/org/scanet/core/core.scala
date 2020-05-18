package org.scanet

import scala.collection.mutable

package object core {

  object syntax extends CoreSyntax

  def error(message: String): Nothing = throw new RuntimeException(message)

  def memoize[I1, O](f: I1 => O): I1 => O = new mutable.HashMap[I1, O]() {self =>
    override def apply(key: I1): O = getOrElseUpdate(key, f(key))
  }

  def memoize[I1, I2, O](f: (I1, I2) => O): (I1, I2) => O = {
    val cached = memoize(f.tupled)
    (i1, i2) => cached((i1, i2))
  }

}
