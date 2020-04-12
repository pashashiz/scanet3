package org.scanet

package object core {

  def error(message: String): Nothing = throw new RuntimeException(message)
}
