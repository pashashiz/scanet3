package scanet

import scala.collection.mutable

package object core {

  object syntax extends CoreSyntax

  def error(message: String): Nothing = throw new RuntimeException(message)

  def memoize[I1, O](f: I1 => O): I1 => O = {
    val cache = mutable.HashMap[I1, O]()
    key => cache.getOrElseUpdate(key, f(key))
  }

  def memoize[I1, I2, O](f: (I1, I2) => O): (I1, I2) => O = {
    val cached = memoize(f.tupled)
    (i1, i2) => cached((i1, i2))
  }

  def memoize[I1, I2, I3, O](f: (I1, I2, I3) => O): (I1, I2, I3) => O = {
    val cached = memoize(f.tupled)
    (i1, i2, i3) => cached((i1, i2, i3))
  }

  def memoize[I1, I2, I3, I4, O](f: (I1, I2, I3, I4) => O): (I1, I2, I3, I4) => O = {
    val cached = memoize(f.tupled)
    (i1, i2, i3, i4) => cached((i1, i2, i3, i4))
  }

  def memoize[I1, I2, I3, I4, I5, O](f: (I1, I2, I3, I4, I5) => O): (I1, I2, I3, I4, I5) => O = {
    val cached = memoize(f.tupled)
    (i1, i2, i3, i4, i5) => cached((i1, i2, i3, i4, i5))
  }
}
