package scanet.math

// NOTE: simulacrum does not support multiple types,
// so there is manual implementation
trait Convertible[A, B] extends Serializable {
  def convert(a: A): B
}

object Convertible {
  def apply[A, B](implicit convertible: Convertible[A, B]): Convertible[A, B] = convertible
}

