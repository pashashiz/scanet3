package org.scanet.core

import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.{BlockingDeque, LinkedBlockingDeque}

import org.scanet.core.Output.Compiled
import org.tensorflow.op.{Scope => NativeScope}
import org.tensorflow.{Graph, Output => NativeOutput, Session => NativeSession, Tensor => NativeTensor}
import simulacrum.typeclass
import org.scanet.core.Session.syntax._

import scala.language.existentials
import scala.util.Try
import scala.collection.JavaConverters._

//import scala.util.Using // use with scala 2.13

case class Runner(session: Session, feed: Map[Output[_], Tensor[_]] = Map()) {

  def feed(elems: (Output[_], Tensor[_])*): Runner = copy(feed = feed ++ Map(elems: _*))

  def evalUnsafe(outs: Seq[Output[_]]): Seq[NativeTensor[_]] = {
    session.eval(outs, feed)
  }

  def evalU[A](value: A)(implicit ce: CanEval[A]): ce.Materialized = ce.eval(this, value)

  // REMOVE

  // NOTE: try HList for this
  // ideally we can get rid of SessionInput and SessionOutput
  // and just make native transformation Product[Output[T]] ~> Product[Tensor[T]]
  def evalX[O: SessionInput, T: SessionOutput](out: O): T = {
    val input: (Seq[Int], Seq[Output[_]]) = out.toInput
    val (size, outputs) = input
    val output = evalUnsafe(outputs)
    SessionOutput[T].fromOutput(size, output)
  }

  def eval[A: TensorType](a: Output[A]): Tensor[A] = {
    val nTensor = session.eval(Seq(a), feed).head.asInstanceOf[NativeTensor[A]]
    Tensor.apply[A](nTensor)
  }

  def eval[A: TensorType, B: TensorType](a: Output[A], b: Output[B]): (Tensor[A], Tensor[B]) = {
    evalX[(Id[Output[A]], Id[Output[B]]), (Id[Tensor[A]], Id[Tensor[B]])]((a, b))
  }

  def eval[A: TensorType, B: TensorType, C: TensorType](a: Output[A], b: Output[B], c: Output[C]): (Tensor[A], Tensor[B], Tensor[C]) = {
    evalX[(Id[Output[A]], Id[Output[B]], Id[Output[C]]), (Id[Tensor[A]], Id[Tensor[B]], Id[Tensor[C]])]((a, b, c))
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
class Session extends AutoCloseable {

  val nGraph = new Graph()
  val nSession = new NativeSession(nGraph)
  var state = SessionState(new NativeScope(nGraph), Map.empty)

  def runner: Runner = Runner(this)

  private def compile(out: Output[_]): NativeOutput[_] = {
    val (updatedState, (_, compiledOp)) = out.findOrCompile(state)
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
      state.cache.get(output.id) match {
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

    implicit def seqIsArg: SeqLike[Seq] = new SeqLike[Seq] {
      override def unit[P](seq: Seq[P]): Seq[P] = seq
      override def asSeq[P](arg: Seq[P]): Seq[P] = arg
    }

    implicit def idIsArg: SeqLike[Id] = new SeqLike[Id] {
      override def unit[P](seq: Seq[P]): Id[P] = seq.head
      override def asSeq[P](arg: Id[P]): Seq[P] = Seq(arg)
    }

    implicit def singleOutputIsSessionInputX[SIn1[_]: SeqLike, A: TensorType]: SessionInput[SIn1[Output[A]]] = {
      (out: SIn1[Output[A]]) => {
        val outs = SeqLike[SIn1].asSeq(out)
        (Seq(outs.size), outs)
      }
    }

    implicit def singleTensorIsSessionOutputX[SOut1[_]: SeqLike, A: TensorType]: SessionOutput[SOut1[Tensor[A]]] = {
      (_: Seq[Int], tensors: Seq[NativeTensor[_]]) => {
        val t1 = tensors.map(nt => Tensor.apply[A](nt.asInstanceOf[NativeTensor[A]]))
        SeqLike[SOut1].unit(t1)
      }
    }

    implicit def tuple2OfOutputsIsSessionInputX[
      SIn1[_]: SeqLike, A1: TensorType,
      SIn2[_]: SeqLike, A2: TensorType]
    : SessionInput[(SIn1[Output[A1]], SIn2[Output[A2]])] =
      (out: (SIn1[Output[A1]], SIn2[Output[A2]])) => {
        val (o1, o2) = (SeqLike[SIn1].asSeq(out._1), SeqLike[SIn2].asSeq(out._2))
        (Seq(o1.size, o2.size), o1 ++: o2)
      }

    implicit def tuple2OfTensorsIsSessionOutputX[
      SOut1[_]: SeqLike, A1: TensorType,
      SOut2[_]: SeqLike, A2: TensorType]
    : SessionOutput[(SOut1[Tensor[A1]], SOut2[Tensor[A2]])] =
      (sizes: Seq[Int], tensors: Seq[NativeTensor[_]]) => {
        val pos0 = 0
        val pos1 = pos0 + sizes(0)
        val pos2 = pos1 + sizes(1)
        val t1 = tensors.slice(pos0, pos1).map(nt => Tensor.apply[A1](nt.asInstanceOf[NativeTensor[A1]]))
        val t2 = tensors.slice(pos1, pos2).map(nt => Tensor.apply[A2](nt.asInstanceOf[NativeTensor[A2]]))
        (SeqLike[SOut1].unit(t1), SeqLike[SOut2].unit(t2))
      }

    implicit def tuple3OfOutputsIsSessionInputX[
      SIn1[_]: SeqLike, A1: TensorType,
      SIn2[_]: SeqLike, A2: TensorType,
      SIn3[_]: SeqLike, A3: TensorType]
    : SessionInput[(SIn1[Output[A1]], SIn2[Output[A2]], SIn3[Output[A3]])] =
      (out: (SIn1[Output[A1]], SIn2[Output[A2]], SIn3[Output[A3]])) => {
        val (o1, o2, o3) = (SeqLike[SIn1].asSeq(out._1), SeqLike[SIn2].asSeq(out._2), SeqLike[SIn3].asSeq(out._3))
        (Seq(o1.size, o2.size, o3.size), o1 ++: o2 ++: o3)
      }

    implicit def tuple3OfTensorsIsSessionOutputX[
      SOut1[_]: SeqLike, A1: TensorType,
      SOut2[_]: SeqLike, A2: TensorType,
      SOut3[_]: SeqLike, A3: TensorType]
    : SessionOutput[(SOut1[Tensor[A1]], SOut2[Tensor[A2]], SOut3[Tensor[A3]])] =
      (sizes: Seq[Int], tensors: Seq[NativeTensor[_]]) => {
        val pos0 = 0
        val pos1 = pos0 + sizes(0)
        val pos2 = pos1 + sizes(1)
        val pos3 = pos2 + sizes(2)
        val t1 = tensors.slice(pos0, pos1).map(nt => Tensor.apply[A1](nt.asInstanceOf[NativeTensor[A1]]))
        val t2 = tensors.slice(pos1, pos2).map(nt => Tensor.apply[A2](nt.asInstanceOf[NativeTensor[A2]]))
        val t3 = tensors.slice(pos2, pos3).map(nt => Tensor.apply[A3](nt.asInstanceOf[NativeTensor[A3]]))
        (SeqLike[SOut1].unit(t1), SeqLike[SOut2].unit(t2), SeqLike[SOut3].unit(t3))
      }

    // NEW

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
          val tensors = evalOutput(runner, out1, value)
          out1.toMaterialized(tensors)
        }
      }

    implicit def canEvalTuple2OfOutputContainers[
      Out1[_], T1: TensorType, Out2[_], T2: TensorType]
    (implicit out1: OutputContainer[Out1], out2: OutputContainer[Out2]) =
      new CanEval[(Out1[T1], Out2[T2])] {
        override type Materialized = (out1.Materialized[T1], out2.Materialized[T2])
        override def eval(runner: Runner, value: (Out1[T1], Out2[T2]))
        : (out1.Materialized[T1], out2.Materialized[T2]) = {
          val tensors1 = evalOutput(runner, out1, value._1)
          val tensors2 = evalOutput(runner, out2, value._2)
          (out1.toMaterialized(tensors1), out2.toMaterialized(tensors2))
        }
      }

    implicit def canEvalTuple3OfOutputContainers[
      Out1[_], T1: TensorType, Out2[_], T2: TensorType, Out3[_], T3: TensorType]
    (implicit out1: OutputContainer[Out1], out2: OutputContainer[Out2], out3: OutputContainer[Out3]) =
      new CanEval[(Out1[T1], Out2[T2], Out3[T3])] {
        override type Materialized = (out1.Materialized[T1], out2.Materialized[T2], out3.Materialized[T3])
        override def eval(runner: Runner, value: (Out1[T1], Out2[T2], Out3[T3]))
        : (out1.Materialized[T1], out2.Materialized[T2], out3.Materialized[T3]) = {
          val tensors1 = evalOutput(runner, out1, value._1)
          val tensors2 = evalOutput(runner, out2, value._2)
          val tensors3 = evalOutput(runner, out3, value._3)
          (out1.toMaterialized(tensors1), out2.toMaterialized(tensors2), out3.toMaterialized(tensors3))
        }
      }

    private def evalOutput[Out[_], T: TensorType]
    (runner: Runner, evidence: OutputContainer[Out], output: Out[T]): Seq[Tensor[T]] = {
      val outputs = evidence.outputToSeq(output)
      runner.evalUnsafe(outputs)
        .map(n => Tensor.apply[T](n.asInstanceOf[NativeTensor[T]]))
    }
  }

  trait Syntax extends Implicits with SeqLike.ToSeqLikeOps with SessionInput.ToSessionInputOps  with SessionOutput.ToSessionOutputOps

  object syntax extends Syntax
}

// REMOVE START
@typeclass trait SeqLike[F[_]] {
  def unit[P](seq: Seq[P]): F[P]
  def asSeq[P](arg: F[P]): Seq[P]
}

@typeclass trait SessionInput[O] {
  def toInput(value: O): (Seq[Int], Seq[Output[_]])
}

@typeclass trait SessionOutput[T] {
  def fromOutput(sizes: Seq[Int], tensors: Seq[NativeTensor[_]]): T
}
// REMOVE END

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