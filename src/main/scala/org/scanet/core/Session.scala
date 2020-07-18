package org.scanet.core

import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.{BlockingDeque, LinkedBlockingDeque}

import org.scanet.core.Output.Compiled
import org.tensorflow.op.{Scope => NativeScope}
import org.tensorflow.{Graph, Output => NativeOutput, Session => NativeSession, Tensor => NativeTensor}

import scala.language.existentials
import scala.util.Try
import scala.collection.JavaConverters._

//import scala.util.Using // use with scala 2.13

case class Runner(session: Session, feed: Map[Output[_], Tensor[_]] = Map()) {

  def feed(elems: (Output[_], Tensor[_])*): Runner = copy(feed = feed ++ Map(elems: _*))

  def evalUnsafe(outs: Seq[Output[_]]): Seq[NativeTensor[_]] = {
    session.eval(outs, feed)
  }

  def eval[A](value: A)(implicit ce: CanEval[A]): ce.Materialized = ce.eval(this, value)
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
class Session extends AutoCloseable {

  val nGraph = new Graph()
  val nSession = new NativeSession(nGraph)
  var state = SessionState(new NativeScope(nGraph), Map.empty)

  def runner: Runner = Runner(this)

  private def compile(out: Output[_]): NativeOutput[_] = {
    val (updatedState, (_, compiledOp)) = out.findOrCompile(state)
    state = updatedState
    compiledOp.output(out.index)
  }

  def toGraph(outs: Seq[Output[_]]): Graph = {
    outs.foreach(out => compile(out))
    nGraph
  }

  private[core] def eval(outs: Seq[Output[_]], feed: Map[Output[_], Tensor[_]]): Seq[NativeTensor[_]] = {
    // with side effect, all compiled options are stored in context cache
    val nativeOutputs = outs.map(out => compile(out))
    val fed = feed.foldLeft(nSession.runner)((runner, entry) => {
      val (output, tensor) = entry
      state.cache.get(output.toString) match {
        case Some((_, output)) =>
          val nativeOutput = output.output(0)
          val nativeTensor = tensor.native
          runner.feed(nativeOutput, nativeTensor)
        case None => runner
      }
    })
    val fetched = nativeOutputs.foldLeft(fed)((runner, output) => runner.fetch(output))
    fetched.run().asScala.toList
  }

  override def close(): Unit = nSession.close()
}

object Session {

  /**
   * Same as:
   * ```
   * Using.resource(new Session()) {
   *   session => ...
   * }
   * ```
   */
  def withing[R](f: Session => R): R = {
    // Using.resource(new Session())(f) // use with scala 2.13
    val session = new Session()
    try {
      val result = f(session)
      session.close()
      result
    } finally if (session != null) session.close()
  }

  trait Implicits {

    implicit def singleOutputIsContainer =
      new OutputContainer[Output] {
        type Materialized[λ] = Tensor[λ]
        override def of[T](seq: Seq[Output[T]]): Output[T] = seq.head
        override def outputToSeq[T](out: Output[T]): Seq[Output[T]] = Seq(out)
        override def materializedToSeq[T](out: Tensor[T]): Seq[Tensor[T]] = Seq(out)
        override def toMaterialized[T](seq: Seq[Tensor[T]]): Tensor[T] = seq.head
      }

    // IMPORTANT: Turned out scala compiler needs OutputSeq[λ] type to be
    // specified explicitly instead of just using Seq[Output[λ]]
    // that is kind of annoying, would be nice to fix it somehow
    // ({type OutputSeq[A] = Seq[Output[A]]})#OutputSeq
    implicit def outputSeqIsContainer =
      new OutputContainer[OutputSeq] {
        override type Materialized[λ] = Seq[Tensor[λ]]
        override def of[T](seq: Seq[Output[T]]): OutputSeq[T] = seq
        override def outputToSeq[T](out: OutputSeq[T]): Seq[Output[T]] = out
        override def materializedToSeq[T](out: Seq[Tensor[T]]): Seq[Tensor[T]] = out
        override def toMaterialized[T](seq: Seq[Tensor[T]]): Seq[Tensor[T]] = seq
      }

    implicit def canEvalOutputContainer[Out1[_], T1: TensorType](implicit out1: OutputContainer[Out1]) =
      new CanEval[Out1[T1]] {

        override type Materialized = out1.Materialized[T1]

        override def eval(runner: Runner, value: Out1[T1]): out1.Materialized[T1] = {
          val outputs = out1.outputToSeq(value)
          val tensors = runner.evalUnsafe(outputs)
            .map(n => Tensor.apply[T1](n.asInstanceOf[NativeTensor[T1]]))
          out1.toMaterialized(tensors)
        }
        override def unwrap(value: Out1[T1]): Seq[Output[_]] = {
          out1.outputToSeq(value)
        }
      }

    implicit def canEvalTuple2OfOutputContainers[
      Out1[_], T1: TensorType, Out2[_], T2: TensorType]
    (implicit out1: OutputContainer[Out1], out2: OutputContainer[Out2]) =
      new CanEval[(Out1[T1], Out2[T2])] {

        override type Materialized = (out1.Materialized[T1], out2.Materialized[T2])

        override def eval(runner: Runner, value: (Out1[T1], Out2[T2]))
        : (out1.Materialized[T1], out2.Materialized[T2]) = {
          val seq1 = out1.outputToSeq(value._1)
          val seq2 = out2.outputToSeq(value._2)
          val results = runner.evalUnsafe(seq1 ++ seq2)
          val tensors1 = results.take(seq1.size)
            .map(n => Tensor.apply[T1](n.asInstanceOf[NativeTensor[T1]]))
          val tensors2 = results.slice(seq1.size, seq1.size + seq2.size)
            .map(n => Tensor.apply[T2](n.asInstanceOf[NativeTensor[T2]]))
          (out1.toMaterialized(tensors1), out2.toMaterialized(tensors2))
        }

        override def unwrap(value: (Out1[T1], Out2[T2])): Seq[Output[_]] = {
          val seq1 = out1.outputToSeq(value._1)
          val seq2 = out2.outputToSeq(value._2)
          seq1 ++ seq2
        }
      }

    implicit def canEvalTuple3OfOutputContainers[
      Out1[_], T1: TensorType, Out2[_], T2: TensorType, Out3[_], T3: TensorType]
    (implicit out1: OutputContainer[Out1], out2: OutputContainer[Out2], out3: OutputContainer[Out3]) =
      new CanEval[(Out1[T1], Out2[T2], Out3[T3])] {

        override type Materialized = (out1.Materialized[T1], out2.Materialized[T2], out3.Materialized[T3])

        override def eval(runner: Runner, value: (Out1[T1], Out2[T2], Out3[T3]))
        : (out1.Materialized[T1], out2.Materialized[T2], out3.Materialized[T3]) = {
          val seq1 = out1.outputToSeq(value._1)
          val seq2 = out2.outputToSeq(value._2)
          val seq3 = out3.outputToSeq(value._3)
          val results = runner.evalUnsafe(seq1 ++ seq2 ++ seq3)
          val tensors1 = results.take(seq1.size)
            .map(n => Tensor.apply[T1](n.asInstanceOf[NativeTensor[T1]]))
          val tensors2 = results.slice(seq1.size, seq1.size + seq2.size)
            .map(n => Tensor.apply[T2](n.asInstanceOf[NativeTensor[T2]]))
          val tensors3 = results.slice(seq1.size + seq2.size, seq1.size + seq2.size + seq3.size)
            .map(n => Tensor.apply[T3](n.asInstanceOf[NativeTensor[T3]]))
          (out1.toMaterialized(tensors1), out2.toMaterialized(tensors2), out3.toMaterialized(tensors3))
        }

        override def unwrap(value: (Out1[T1], Out2[T2], Out3[T3])): Seq[Output[_]] = {
          val seq1 = out1.outputToSeq(value._1)
          val seq2 = out2.outputToSeq(value._2)
          val seq3 = out3.outputToSeq(value._3)
          seq1 ++ seq2 ++ seq3
        }
      }
  }

  trait Syntax extends Implicits

  object syntax extends Syntax
}

trait OutputContainer[O[_]] {
  type Materialized[_]
  def of[T](seq: Seq[Output[T]]): O[T]
  def outputToSeq[T](out: O[T]): Seq[Output[T]]
  def materializedToSeq[T](out: Materialized[T]): Seq[Tensor[T]]
  def toMaterialized[T](seq: Seq[Tensor[T]]): Materialized[T]
}

trait CanEval[In] {
  type Materialized
  def eval(runner: Runner, value: In): Materialized
  def unwrap(value: In): Seq[Output[_]]
}

class LazySession {
  lazy val get = new Session()
}

class SessionPool(val maxSize: Int) {

  private val pool: BlockingDeque[LazySession] = new LinkedBlockingDeque[LazySession](maxSize)
  private val inUse: AtomicInteger = new AtomicInteger(0)

  pool.addAll(List.fill(maxSize)(new LazySession()).asJavaCollection)

  def used: Int = inUse.get

  def withing[R](f: Session => R): R = {
    val session = pool.takeFirst()
    inUse.incrementAndGet()
    val result = Try {f(session.get)}
    pool.addFirst(session)
    inUse.decrementAndGet()
    result.get
  }
}

object SessionPool {
  def max(size: Int): SessionPool = new SessionPool(size)
}