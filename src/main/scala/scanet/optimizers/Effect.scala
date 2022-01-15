package scanet.optimizers

import org.apache.spark.sql.Dataset
import scanet.core.{Convertible, Floating, Numeric, TensorBoard}
import scanet.estimators._
import scanet.math.syntax._

trait Effect[E] extends (E => Unit) {
  private val self = this
  def apply(next: E): Unit
  def onClose(): Unit = {}
  def conditional(cond: E => Boolean): Effect[E] = new Effect[E] {
    override def apply(next: E) = if (cond(next)) self(next)
    override def onClose() = self.onClose()
  }
}

object Effect {

  case class RecordLoss[E: Numeric](
      console: Boolean = true,
      tensorboard: Boolean = false,
      dir: String = "board")(implicit c: Convertible[E, Float])
      extends Effect[StepContext[E]] {
    val board = TensorBoard(dir)
    override def apply(ctx: StepContext[E]) = {
      val loss = ctx.result.loss
      if (console)
        println(s"#${ctx.step.epoch}:${ctx.step.iter} loss: $loss")
      if (tensorboard)
        board.addScalar("loss", loss, ctx.step.iter)
    }
  }

  case class RecordAccuracy[E: Floating](
      ds: Dataset[Record[E]],
      console: Boolean = true,
      tensorboard: Boolean = false,
      dir: String = "board")(implicit c: Convertible[E, Float])
      extends Effect[StepContext[E]] {
    val board = TensorBoard(dir)
    override def apply(ctx: StepContext[E]) = {
      val trained = ctx.lossModel.trained(ctx.result.weights)
      val a = accuracy(trained, ds)
      if (console)
        println(s"accuracy: $a")
      if (tensorboard)
        board.addScalar("accuracy", a, ctx.step.iter)
    }
  }
}