package scanet.optimizers

import org.apache.spark.sql.Dataset
import scanet.core.{Convertible, Floating, Numeric, TensorBoard}
import scanet.estimators._
import scanet.math.syntax._
import scanet.optimizers.Effect.State

trait Effect[E] extends ((State, StepContext[E]) => State) {
  private val self = this
  def apply(state: State, next: StepContext[E]): State
  def conditional(cond: StepContext[E] => Boolean): Effect[E] =
    (state: State, next: StepContext[E]) => {
      if (cond(next)) self(state, next) else state
    }
}

object Effect {

  case class Console(events: Seq[String] = Nil) {
    def write(event: String): Console = Console(events :+ event)
  }

  case class State(console: Console, board: TensorBoard) {
    def write(event: String): State = copy(console = console.write(event))
    def writeIf(cond: Boolean, event: String): State =
      if (cond) copy(console = console.write(event)) else this
    def run(): Unit = {
      println(console.events.mkString(", "))
    }
  }

  case class RecordIteration[E: Numeric]() extends Effect[E] {
    override def apply(state: State, next: StepContext[E]): State = {
      val timeSec = next.time.toDouble / 1000
      state.write(s"${next.step.epoch}/${next.step.iter} ${timeSec}s")
    }
  }

  case class RecordLoss[E: Numeric](
      console: Boolean = true,
      tensorboard: Boolean = false)(implicit c: Convertible[E, Float])
      extends Effect[E] {
    override def apply(state: State, next: StepContext[E]): State = {
      val loss = next.result.loss
      if (tensorboard)
        state.board.addScalar("loss", loss, next.step.iter)
      state.writeIf(console, s"loss: $loss")
    }
  }

  case class RecordAccuracy[E: Floating](
      ds: Dataset[Record[E]],
      console: Boolean = true,
      tensorboard: Boolean = false)
      extends Effect[E] {
    override def apply(state: State, next: StepContext[E]): State = {
      val trained = next.lossModel.trained(next.result.params)
      val a = accuracy(trained, ds, next.step.batch)
      if (tensorboard)
        state.board.addScalar("accuracy", a, next.step.iter)
      state.writeIf(console, s"accuracy: $a")
    }
  }
}
