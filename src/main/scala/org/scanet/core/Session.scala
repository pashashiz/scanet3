package org.scanet.core

import org.scanet.core.Output.Compiled
import org.tensorflow.op.{Scope => NativeScope}
import org.tensorflow.{Graph, Output => NativeOutput, Session => NativeSession, Tensor => NativeTensor}

import scala.jdk.CollectionConverters._
import scala.language.existentials

case class Runner(session: Session, feed: Map[Output[_], Tensor[_]] = Map()) {

  def feed(elems: (Output[_], Tensor[_])*): Runner = copy(feed = Map(elems: _*))

  def eval[A: TensorType](a: Output[A]): Tensor[A] = {
    val nTensor = session.eval(Seq(a), feed).head.asInstanceOf[NativeTensor[A]]
    Tensor.apply[A](nTensor)
  }

  def eval[A: TensorType, B: TensorType](a: Output[A], b: Output[B]): (Tensor[A], Tensor[B]) = {
    val tensors = session.eval(Seq(a, b), feed)
    (Tensor.apply[A](tensors(0).asInstanceOf[NativeTensor[A]]),
      Tensor.apply[B](tensors(1).asInstanceOf[NativeTensor[B]]))
  }

  def eval[A: TensorType, B: TensorType, C: TensorType](a: Output[A], b: Output[B], c: Output[C]): (Tensor[A], Tensor[B], Tensor[C]) = {
    val tensors = session.eval(Seq(a, b, c), feed)
    (Tensor.apply[A](tensors(0).asInstanceOf[NativeTensor[A]]),
      Tensor.apply[B](tensors(1).asInstanceOf[NativeTensor[B]]),
      Tensor.apply[C](tensors(2).asInstanceOf[NativeTensor[C]]))
  }
}

case class SessionState(scope: NativeScope, cache: Map[String, Compiled]) {
  def maxLabelIndex(name: String): Int = {
    // NOTE: in the future think about prebuilt index
    val names = cache.values.map(_._1).groupBy(_.value)
    names.get(name).map(n => n.map(_.index).max).getOrElse(-1)
  }
}

/** Session is a mutable object which keeps track of all compiled operations
  * and may reuse them if executed again. Sometimes that iss important to use the same session
  * when we need to run the same execution graph over and over by changing placeholders,
  * such graph would be constructed and optimized only once.
  *
  * Recommended usage withing `using` boundary which will release native resources:
  * {{{
  * using(session => {
 *   val a = placeholder[Int]()
 *   val b = 10.const
 *   val c = a + b
 *   session.runner
 *     .feed(a -> Tensor.scalar(5))
 *     .eval(c) should be(Tensor.scalar(15))
 * })
  * }}}
  */
class Session {

  val nGraph = new Graph()
  val nSession = new NativeSession(nGraph)
  var state = SessionState(new NativeScope(nGraph), Map.empty)

  def runner: Runner = Runner(this)

  private def compile(out: Output[_]): NativeOutput[_] = {
    val (updatedState, (_, compiledOp)) = out.compile(state)
    state = updatedState
    compiledOp.output(0)
  }

  private[core] def toGraph(outs: Seq[Output[_]]): Graph = {
    outs.foreach(out => compile(out))
    nGraph
  }

  private[core] def eval(outs: Seq[Output[_]], feed: Map[Output[_], Tensor[_]]): Seq[NativeTensor[_]] = {
    // with side effect, all compiled options are stored in context cache
    val nativeOutputs = outs.map(out => compile(out))
    val fed = feed.foldLeft(nSession.runner)((runner, entry) => {
      val (output, tensor) = entry
      val nativeOutput = state.cache(output.id)._2.output(0)
      val nativeTensor = tensor.native.asInstanceOf[NativeTensor[_]]
      runner.feed(nativeOutput, nativeTensor)
    })
    val fetched = nativeOutputs.foldLeft(fed)((runner, output) => runner.fetch(output))
    fetched.run().asScala.toList
  }

  def close(): Unit = nSession.close()

}

object Session {

  def using[R](f: Session => R): R = {
    val session = new Session()
    try {
      val result = f(session)
      session.close()
      result
    } finally if (session != null) session.close()
  }
}
