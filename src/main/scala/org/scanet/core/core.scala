package org.scanet

package object core {

  object syntax extends CoreSyntax

  def error(message: String): Nothing = throw new RuntimeException(message)

}
