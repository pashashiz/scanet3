package org.scanet.optimizers

import org.scanet.core._
import org.scanet.math.{Convertible, Numeric}

case class Effect[S, E](zero: S, action: (S, E) => S) {
  def conditional(cond: E => Boolean): Effect[S, E] = {
    copy(action = (a, s) => {
      if (cond(s)) action(a, s) else a
    })
  }
}

object Effect {

  def stateless[E](action: E => Unit): Effect[Unit, E] =
    Effect((), (_, element) => action(element))

  def plotResult[W: Numeric: TensorType, R: Numeric: TensorType]
    (name: String = "result", dir: String = "board")
    (implicit c: Convertible[R, Float]): Effect[TensorBoard, Step[W, R]] = {
    Effect(new TensorBoard(dir), (board, step) => {
      board.addScalar(name, step.result.get, step.iter)
    })
  }

  def logResult[W: Numeric: TensorType, R: Numeric: TensorType]()
    (implicit c: Convertible[R, Float]): Effect[Unit, Step[W, R]] =
    Effect.stateless(step => println(
      s"#${step.epoch}:${step.iter} loss: ${step.result.map(_.toString).getOrElse("")}"))
}

case class Effects[S](all: Seq[Effect[_, S]]) {
  def unit: Seq[_] = all.map(_.zero)
  def action(acc: Seq[_], next: S): Seq[_] = acc.zip(all) map {
    case (a, Effect(_, action)) => action(a, next)
  }
  def :+ (effect: Effect[_, S]): Effects[S] = Effects(all :+ effect)
}

object Effects {

  def empty[S]: Effects[S] = Effects(Seq())
}