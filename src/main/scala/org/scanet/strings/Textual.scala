package org.scanet.strings

trait Textual[S]

object Textual {

  trait Syntax {
    implicit def stringIsTextual: Textual[String] = new Textual[String] {}
  }

  object syntax extends Syntax
}
