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
import TensorType.syntax._
import ConstOp.syntax._
import Slice.syntax._

//import scala.util.Using // use with scala 2.13

case class Runner(session: Session, feed: Map[Output[_], Tensor[_]] = Map()) {

  def feed(elems: (Output[_], Tensor[_])*): Runner = copy(feed = Map(elems: _*))

  def evalUnsafe(outs: Seq[Output[_]]): Seq[NativeTensor[_]] = {
    session.eval(outs, feed)
  }

  // NOTE: try HList for this
  // ideally we can get rid of SessionInput and SessionOutput
  // and just make native transformation Product[Output[T]] ~> Product[Tensor[T]]
  def evalX[O: SessionInput, T: SessionOutput](out: O): T = {
    val input: Seq[Output[_]] = out.toInput
    val output = evalUnsafe(input)
    SessionOutput[T].fromOutput(output)
  }

  def evalXX[O: SessionInputX, T: SessionOutputX](out: O): T = {
    import org.scanet.core.Session.syntaxX._
    val input: Seq[Output[_]] = out.toInput
    val output = evalUnsafe(input)
    SessionOutputX[T].fromOutput(output)
  }

  def eval[A: TensorType](a: Output[A]): Tensor[A] = {
    val nTensor = session.eval(Seq(a), feed).head.asInstanceOf[NativeTensor[A]]
    Tensor.apply[A](nTensor)
  }

  def eval[A: TensorType, B: TensorType](a: Output[A], b: Output[B]): (Tensor[A], Tensor[B]) = {
    evalX[(Output[A], Output[B]), (Tensor[A], Tensor[B])]((a, b))
  }

  def eval[A: TensorType, B: TensorType, C: TensorType](a: Output[A], b: Output[B], c: Output[C]): (Tensor[A], Tensor[B], Tensor[C]) = {
    evalX[(Output[A], Output[B], Output[C]), (Tensor[A], Tensor[B], Tensor[C])]((a, b, c))
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

  // todo: remove
  trait Implicits {

    implicit def singleOutputIsSessionInput[A: TensorType]: SessionInput[Output[A]] =
      (out: Output[A]) => Seq(out)

    implicit def singleTensorIsSessionOutput[A: TensorType]: SessionOutput[Tensor[A]] =
      (tensors: Seq[NativeTensor[_]]) => Tensor.apply[A](tensors(0).asInstanceOf[NativeTensor[A]])

    implicit def tuple2OfOutputsIsSessionInput[A1: TensorType, A2: TensorType]: SessionInput[(Output[A1], Output[A2])] =
      (out: (Output[A1], Output[A2])) => Seq(out._1, out._2)

    implicit def tuple2OfTensorsIsSessionOutput[A1: TensorType, A2: TensorType]: SessionOutput[(Tensor[A1], Tensor[A2])] =
      (tensors: Seq[NativeTensor[_]]) => (
        Tensor.apply[A1](tensors(0).asInstanceOf[NativeTensor[A1]]),
        Tensor.apply[A2](tensors(1).asInstanceOf[NativeTensor[A2]])
      )

    implicit def tuple3OfOutputsIsSessionInput[A1: TensorType, A2: TensorType, A3: TensorType]: SessionInput[(Output[A1], Output[A2], Output[A3])] =
      (out: (Output[A1], Output[A2], Output[A3])) => Seq(out._1, out._2, out._3)

    implicit def tuple3OfTensorsIsSessionOutput[A1: TensorType, A2: TensorType, A3: TensorType]: SessionOutput[(Tensor[A1], Tensor[A2], Tensor[A3])] =
      (tensors: Seq[NativeTensor[_]]) => (
        Tensor.apply[A1](tensors(0).asInstanceOf[NativeTensor[A1]]),
        Tensor.apply[A2](tensors(1).asInstanceOf[NativeTensor[A2]]),
        Tensor.apply[A3](tensors(2).asInstanceOf[NativeTensor[A3]])
      )

  }

  trait ImplicitsX {

    implicit def seqIsArg: SeqLike[Seq] = new SeqLike[Seq] {
      override def unit[P](seq: Seq[P]): Seq[P] = seq
      override def asSeq[P](arg: Seq[P]): Seq[P] = arg
    }

    implicit def idIsArg: SeqLike[Id] = new SeqLike[Id] {
      override def unit[P](seq: Seq[P]): Id[P] = seq.head
      override def asSeq[P](arg: Id[P]): Seq[P] = Seq(arg)
    }

    implicit def singleOutputIsSessionInputX[SIn1[_]: SeqLike, A: TensorType]: SessionInputX[SIn1[Output[A]]] = {
      (out: SIn1[Output[A]]) => {
        import SeqLike.ops._
        val outs = out.asSeq
        Tensor.vector(outs.size).const +: outs
      }
    }

    implicit def singleTensorIsSessionOutputX[SOut1[_]: SeqLike, A: TensorType]: SessionOutputX[SOut1[Tensor[A]]] = {
      (tensors: Seq[NativeTensor[_]]) => {
        val size = Tensor.apply[Int](tensors(0).asInstanceOf[NativeTensor[Int]]).slice(0).toScalar
        val converted = tensors.slice(1, size + 1).map(nt => Tensor.apply[A](nt.asInstanceOf[NativeTensor[A]]))
        SeqLike[SOut1].unit(converted)
      }
    }
  }

  trait Syntax extends Implicits with SessionInput.ToSessionInputOps

  trait SyntaxX extends ImplicitsX with SeqLike.ToSeqLikeOps with SessionInputX.ToSessionInputXOps with SessionOutputX.ToSessionOutputXOps

  object syntax extends Syntax

  object syntaxX extends SyntaxX
}

// todo: remove
@typeclass trait SessionInput[O] {
  def toInput(value: O): Seq[Output[_]]
}
// todo: remove
@typeclass trait SessionOutput[T] {
  def fromOutput(tensors: Seq[NativeTensor[_]]): T
}

@typeclass trait SeqLike[F[_]] {
  def unit[P](seq: Seq[P]): F[P]
  def asSeq[P](arg: F[P]): Seq[P]
}

@typeclass trait SessionInputX[O] {
  def toInput(value: O): Seq[Output[_]]
}

@typeclass trait SessionOutputX[T] {
  def fromOutput(tensors: Seq[NativeTensor[_]]): T
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