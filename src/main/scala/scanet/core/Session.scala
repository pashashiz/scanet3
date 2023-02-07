package scanet.core

import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.{BlockingDeque, LinkedBlockingDeque}
import org.tensorflow.op.{OpScope, Scope => NativeScope}
import org.tensorflow.{Graph, RawTensor, Output => NativeOutput, Session => NativeSession}

import scala.util.Try
import scala.jdk.CollectionConverters._
import scala.collection.immutable.Seq

case class Runner(session: Session, feed: Map[Expr[_], Tensor[_]] = Map()) {

  def feed(elems: (Expr[_], Tensor[_])*): Runner = copy(feed = feed ++ Map(elems: _*))

  def evalUnsafe(outs: Seq[Expr[_]]): Seq[RawTensor] = {
    session.eval(outs, feed)
  }

  def eval[A](value: A)(implicit m: Mat[A]): m.Out = {
    val (layout, allExpr) = m.deconstructIn(value)
    m.constructOutRaw(layout, evalUnsafe(allExpr))
  }
}

case class SessionState(scope: NativeScope, cache: Map[String, LabeledOperationOut]) {
  def maxLabelIndex(name: String): Int = {
    // NOTE: in the future think about prebuilt index
    val names = cache.values.map(_.label).groupBy(_.name)
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
  var state = SessionState(new OpScope(nGraph), Map.empty)

  def runner: Runner = Runner(this)

  private def compile(out: Expr[_]): NativeOutput[_] = {
    val (updatedState, labeledOut) = out.compile(state)
    state = updatedState
    labeledOut.operationOut.outputOrFail
  }

  def toGraph(outs: Seq[Expr[_]]): Graph = {
    outs.foreach(out => compile(out))
    nGraph
  }

  private[core] def eval(outs: Seq[Expr[_]], feed: Map[Expr[_], Tensor[_]]): Seq[RawTensor] = {
    // we might have multiple duplicated outputs, we would like to eval only unique
    val outputs = Outputs(outs)
    // with side effect, all compiled options are stored in context cache
    val nativeOutputs = outputs.compress.map(out => compile(out))
    val fed = feed.foldLeft(nSession.runner)((runner, entry) => {
      val (output, tensor) = entry
      state.cache.get(output.ref) match {
        case Some(LabeledOperationOut(operationOut, _)) =>
          val nativeOutput = operationOut.outputOrFail
          val nativeTensor = tensor.native
          runner.feed(nativeOutput, nativeTensor)
        case None => runner
      }
    })
    val fetched = nativeOutputs.foldLeft(fed)((runner, output) => runner.fetch(output))
    val outTensors = fetched.run().asScala.map(_.getValue.asRawTensor()).toList
    outputs.uncompress(outTensors)
  }

  override def close(): Unit = nSession.close()
}

case class Outputs(original: Seq[Expr[_]]) {
  private val uniqueIndex = original.zipWithIndex
    .groupBy { case (out, _) => out.ref }.toList.map {
      case (_, all) =>
        val (out, _) = all.head
        val indexes = all.map { case (_, originalIndex) => originalIndex }
        (out, indexes)
    }
  private val reverseIndex = uniqueIndex.zipWithIndex
    .flatMap {
      case ((_, originalIndexes), uniqueIndex) =>
        originalIndexes.map(index => (index, uniqueIndex))
    }
    .sortBy { case (originalIndex, _) => originalIndex }
    .map { case (_, uniqueIndex) => uniqueIndex }
  val compress: Seq[Expr[_]] = uniqueIndex.map(_._1)
  def uncompress[A](unique: Seq[A]): Seq[A] = reverseIndex.map(unique)
}

object Session {

  /** Same as:
    * {{{
    *   Using.resource(new Session()) {
    *   session => ...
    * }}}
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
}

class LazySession {
  lazy val get = new Session()
}

class SessionPool(val maxSize: Int) {

  private val pool: BlockingDeque[LazySession] = new LinkedBlockingDeque[LazySession](maxSize)
  private val inUse: AtomicInteger = new AtomicInteger(0)

  pool.addAll(List.fill(maxSize)(new LazySession()).asJavaCollection)

  def used: Int = inUse.get

  def within[R](f: Session => R): R = {
    val session = pool.takeFirst()
    inUse.incrementAndGet()
    val result = Try { f(session.get) }
    pool.addFirst(session)
    inUse.decrementAndGet()
    result.get
  }
}

object SessionPool {
  def max(size: Int): SessionPool = new SessionPool(size)
}
