package scanet.core

import scanet.math.syntax._

import scala.collection.concurrent
import scala.collection.immutable.Seq

object TF {

  class TF1Ops[A1, R](f: A1 => R) {
    def tf: TF1[A1, R] = TF1(f)
  }

  class TF2Ops[A1, A2, R](f: (A1, A2) => R) {
    def tf: TF2[A1, A2, R] = TF2(f)
  }

  class TF3Ops[A1, A2, A3, R](f: (A1, A2, A3) => R) {
    def tf: TF3[A1, A2, A3, R] = TF3(f)
  }

  class TF4Ops[A1, A2, A3, A4, R](f: (A1, A2, A3, A4) => R) {
    def tf: TF4[A1, A2, A3, A4, R] = TF4(f)
  }

  class TF5Ops[A1, A2, A3, A4, A5, R](f: (A1, A2, A3, A4, A5) => R) {
    def tf: TF5[A1, A2, A3, A4, A5, R] = TF5(f)
  }

  trait AllSyntax {

    // implicit conversions to TF when, useful when we want to compile regular function
    implicit def toTF1[A1, R](f: A1 => R): TF1[A1, R] = TF1(f)
    implicit def toTF2[A1, A2, R](f: (A1, A2) => R): TF2[A1, A2, R] = TF2(f)
    implicit def toTF3[A1, A2, A3, R](f: (A1, A2, A3) => R): TF3[A1, A2, A3, R] = TF3(f)
    implicit def toTF4[A1, A2, A3, A4, R](f: (A1, A2, A3, A4) => R): TF4[A1, A2, A3, A4, R] = TF4(f)
    implicit def toTF5[A1, A2, A3, A4, A5, R](f: (A1, A2, A3, A4, A5) => R)
        : TF5[A1, A2, A3, A4, A5, R] = TF5(f)

    // ops to by calling f.tf we can convert a function to FT on demand
    implicit def toTF1Ops[A1, R](f: A1 => R): TF1Ops[A1, R] = new TF1Ops(f)
    implicit def toTF2Ops[A1, A2, R](f: (A1, A2) => R): TF2Ops[A1, A2, R] = new TF2Ops(f)
    implicit def toTF3Ops[A1, A2, A3, R](f: (A1, A2, A3) => R): TF3Ops[A1, A2, A3, R] =
      new TF3Ops(f)
    implicit def toTF4Ops[A1, A2, A3, A4, R](f: (A1, A2, A3, A4) => R): TF4Ops[A1, A2, A3, A4, R] =
      new TF4Ops(f)
    implicit def toTF5Ops[A1, A2, A3, A4, A5, R](f: (A1, A2, A3, A4, A5) => R)
        : TF5Ops[A1, A2, A3, A4, A5, R] = new TF5Ops(f)

  }

  object syntax extends AllSyntax

  case class Cache[R]() {
    private val map = concurrent.TrieMap[Seq[Shape], (Seq[Expr[_]], R)]()
    def getOrCompute(key: Seq[Shape])(op: => (Seq[Expr[_]], R)): (Seq[Expr[_]], R) = {
      map.get(key) match {
        case Some(v) => v
        case None    => val d = op; map(key) = d; d
      }
    }
  }
}

trait TF1[A1, R] {
  def compileWith(session: Session)(implicit a1Mat: Mat[A1], rMat: Mat[R]): a1Mat.Out => rMat.Out
  def compile(implicit a1Mat: Mat[A1], rMat: Mat[R]): a1Mat.Out => rMat.Out =
    compileWith(new Session())(a1Mat, rMat)
}

object TF1 {

  def apply[A1, R](func: A1 => R): TF1[A1, R] = new TF1Cached[A1, R](func)

  class TF1Cached[A1, R](func: A1 => R) extends TF1[A1, R] {

    private val cache = TF.Cache[R]()

    override def compileWith(session: Session)(
        implicit a1Mat: Mat[A1],
        rMat: Mat[R]): a1Mat.Out => rMat.Out = {
      a1Out: a1Mat.Out =>
        {
          val (a1Layout, a1T) = a1Mat.deconstructOut(a1Out)
          val a1Type = a1T.map(tensor => (tensor.`type`, tensor.shape))

          // when function is run multiple times with tensor arguments of the same shape
          // we can reuse the same Expr graph, we have built once
          // it will allow the session to reuse the Expr graph (since it remembers all)
          // and compile it only once to native code
          val (p1, r) = cache.getOrCompute(a1Type.map(_._2)) {
            val p1 = a1Type.map { case (t, s) => placeholderRaw(t, s) }
            val a1 = a1Mat.constructIn(a1Layout, p1)
            val r = func(a1)
            (p1, r)
          }

          val (rLayout, rIn) = rMat.deconstructIn(r)
          val rRaw = session.runner.feed(p1 zip a1T: _*).evalUnsafe(rIn)
          rMat.constructOutRaw(rLayout, rRaw)
        }
    }
  }
}

trait TF2[A1, A2, R] {
  def compileWith(session: Session)(
      implicit a1Mat: Mat[A1],
      a2Mat: Mat[A2],
      rMat: Mat[R]): (a1Mat.Out, a2Mat.Out) => rMat.Out
  def compile(
      implicit a1Mat: Mat[A1],
      a2Mat: Mat[A2],
      rMat: Mat[R]): (a1Mat.Out, a2Mat.Out) => rMat.Out =
    compileWith(new Session())(a1Mat, a2Mat, rMat)
}

object TF2 {

  def apply[A1, A2, R](func: (A1, A2) => R): TF2[A1, A2, R] = new TF2Cached[A1, A2, R](func)

  class TF2Cached[A1, A2, R](func: (A1, A2) => R) extends TF2[A1, A2, R] {

    private val cache = TF.Cache[R]()

    override def compileWith(session: Session)(
        implicit a1Mat: Mat[A1],
        a2Mat: Mat[A2],
        rMat: Mat[R]): (a1Mat.Out, a2Mat.Out) => rMat.Out = {
      (a1Out: a1Mat.Out, a2Out: a2Mat.Out) =>
        {
          val (a1Layout, a1T) = a1Mat.deconstructOut(a1Out)
          val (a2Layout, a2T) = a2Mat.deconstructOut(a2Out)
          val aTAll = a1T ++ a2T
          val a1Type = a1T.map(tensor => (tensor.`type`, tensor.shape))
          val a2Type = a2T.map(tensor => (tensor.`type`, tensor.shape))
          val aShapes = a1Type.map(_._2) ++ a2Type.map(_._2)

          val (pAll, r) = cache.getOrCompute(aShapes) {
            val p1 = a1Type.map { case (t, s) => placeholderRaw(t, s) }
            val p2 = a2Type.map { case (t, s) => placeholderRaw(t, s) }
            val a1 = a1Mat.constructIn(a1Layout, p1)
            val a2 = a2Mat.constructIn(a2Layout, p2)
            val r = func(a1, a2)
            (p1 ++ p2, r)
          }

          val (rLayout, rIn) = rMat.deconstructIn(r)
          val rRaw = session.runner.feed(pAll zip aTAll: _*).evalUnsafe(rIn)
          rMat.constructOutRaw(rLayout, rRaw)
        }
    }
  }
}

trait TF3[A1, A2, A3, R] {
  def compileWith(session: Session)(
      implicit a1Mat: Mat[A1],
      a2Mat: Mat[A2],
      a3Mat: Mat[A3],
      rMat: Mat[R]): (a1Mat.Out, a2Mat.Out, a3Mat.Out) => rMat.Out
  def compile(
      implicit a1Mat: Mat[A1],
      a2Mat: Mat[A2],
      a3Mat: Mat[A3],
      rMat: Mat[R]): (a1Mat.Out, a2Mat.Out, a3Mat.Out) => rMat.Out =
    compileWith(new Session())(a1Mat, a2Mat, a3Mat, rMat)
}

object TF3 {

  def apply[A1, A2, A3, R](func: (A1, A2, A3) => R): TF3[A1, A2, A3, R] =
    new TF3Cached[A1, A2, A3, R](func)

  class TF3Cached[A1, A2, A3, R](func: (A1, A2, A3) => R) extends TF3[A1, A2, A3, R] {

    private val cache = TF.Cache[R]()

    override def compileWith(session: Session)(
        implicit a1Mat: Mat[A1],
        a2Mat: Mat[A2],
        a3Mat: Mat[A3],
        rMat: Mat[R]): (a1Mat.Out, a2Mat.Out, a3Mat.Out) => rMat.Out = {
      (a1Out: a1Mat.Out, a2Out: a2Mat.Out, a3Out: a3Mat.Out) =>
        {
          val (a1Layout, a1T) = a1Mat.deconstructOut(a1Out)
          val (a2Layout, a2T) = a2Mat.deconstructOut(a2Out)
          val (a3Layout, a3T) = a3Mat.deconstructOut(a3Out)
          val aTAll = a1T ++ a2T ++ a3T
          val a1Type = a1T.map(tensor => (tensor.`type`, tensor.shape))
          val a2Type = a2T.map(tensor => (tensor.`type`, tensor.shape))
          val a3Type = a3T.map(tensor => (tensor.`type`, tensor.shape))
          val aShapes = a1Type.map(_._2) ++ a2Type.map(_._2) ++ a3Type.map(_._2)

          val (pAll, r) = cache.getOrCompute(aShapes) {
            val p1 = a1Type.map { case (t, s) => placeholderRaw(t, s) }
            val p2 = a2Type.map { case (t, s) => placeholderRaw(t, s) }
            val p3 = a3Type.map { case (t, s) => placeholderRaw(t, s) }
            val a1 = a1Mat.constructIn(a1Layout, p1)
            val a2 = a2Mat.constructIn(a2Layout, p2)
            val a3 = a3Mat.constructIn(a3Layout, p3)
            val r = func(a1, a2, a3)
            (p1 ++ p2 ++ p3, r)
          }

          val (rLayout, rIn) = rMat.deconstructIn(r)
          val rRaw = session.runner.feed(pAll zip aTAll: _*).evalUnsafe(rIn)
          rMat.constructOutRaw(rLayout, rRaw)
        }
    }
  }
}

trait TF4[A1, A2, A3, A4, R] {
  def compileWith(session: Session)(
      implicit a1Mat: Mat[A1],
      a2Mat: Mat[A2],
      a3Mat: Mat[A3],
      a4Mat: Mat[A4],
      rMat: Mat[R]): (a1Mat.Out, a2Mat.Out, a3Mat.Out, a4Mat.Out) => rMat.Out
  def compile(
      implicit a1Mat: Mat[A1],
      a2Mat: Mat[A2],
      a3Mat: Mat[A3],
      a4Mat: Mat[A4],
      rMat: Mat[R]): (a1Mat.Out, a2Mat.Out, a3Mat.Out, a4Mat.Out) => rMat.Out =
    compileWith(new Session())(a1Mat, a2Mat, a3Mat, a4Mat, rMat)
}

object TF4 {

  def apply[A1, A2, A3, A4, R](func: (A1, A2, A3, A4) => R): TF4[A1, A2, A3, A4, R] =
    new TF4Cached[A1, A2, A3, A4, R](func)

  class TF4Cached[A1, A2, A3, A4, R](func: (A1, A2, A3, A4) => R) extends TF4[A1, A2, A3, A4, R] {

    private val cache = TF.Cache[R]()

    override def compileWith(session: Session)(
        implicit a1Mat: Mat[A1],
        a2Mat: Mat[A2],
        a3Mat: Mat[A3],
        a4Mat: Mat[A4],
        rMat: Mat[R]): (a1Mat.Out, a2Mat.Out, a3Mat.Out, a4Mat.Out) => rMat.Out = {
      (a1Out: a1Mat.Out, a2Out: a2Mat.Out, a3Out: a3Mat.Out, a4Out: a4Mat.Out) =>
        {
          val (a1Layout, a1T) = a1Mat.deconstructOut(a1Out)
          val (a2Layout, a2T) = a2Mat.deconstructOut(a2Out)
          val (a3Layout, a3T) = a3Mat.deconstructOut(a3Out)
          val (a4Layout, a4T) = a4Mat.deconstructOut(a4Out)
          val aTAll = a1T ++ a2T ++ a3T ++ a4T
          val a1Type = a1T.map(tensor => (tensor.`type`, tensor.shape))
          val a2Type = a2T.map(tensor => (tensor.`type`, tensor.shape))
          val a3Type = a3T.map(tensor => (tensor.`type`, tensor.shape))
          val a4Type = a4T.map(tensor => (tensor.`type`, tensor.shape))
          val aShapes = a1Type.map(_._2) ++ a2Type.map(_._2) ++ a3Type.map(_._2) ++ a4Type.map(_._2)

          val (pAll, r) = cache.getOrCompute(aShapes) {
            val p1 = a1Type.map { case (t, s) => placeholderRaw(t, s) }
            val p2 = a2Type.map { case (t, s) => placeholderRaw(t, s) }
            val p3 = a3Type.map { case (t, s) => placeholderRaw(t, s) }
            val p4 = a4Type.map { case (t, s) => placeholderRaw(t, s) }
            val a1 = a1Mat.constructIn(a1Layout, p1)
            val a2 = a2Mat.constructIn(a2Layout, p2)
            val a3 = a3Mat.constructIn(a3Layout, p3)
            val a4 = a4Mat.constructIn(a4Layout, p4)
            val r = func(a1, a2, a3, a4)
            (p1 ++ p2 ++ p3 ++ p4, r)
          }

          val (rLayout, rIn) = rMat.deconstructIn(r)
          val rRaw = session.runner.feed(pAll zip aTAll: _*).evalUnsafe(rIn)
          rMat.constructOutRaw(rLayout, rRaw)
        }
    }
  }
}

trait TF5[A1, A2, A3, A4, A5, R] {
  def compileWith(session: Session)(
      implicit a1Mat: Mat[A1],
      a2Mat: Mat[A2],
      a3Mat: Mat[A3],
      a4Mat: Mat[A4],
      a5Mat: Mat[A5],
      rMat: Mat[R]): (a1Mat.Out, a2Mat.Out, a3Mat.Out, a4Mat.Out, a5Mat.Out) => rMat.Out
  def compile(
      implicit a1Mat: Mat[A1],
      a2Mat: Mat[A2],
      a3Mat: Mat[A3],
      a4Mat: Mat[A4],
      a5Mat: Mat[A5],
      rMat: Mat[R]): (a1Mat.Out, a2Mat.Out, a3Mat.Out, a4Mat.Out, a5Mat.Out) => rMat.Out =
    compileWith(new Session())(a1Mat, a2Mat, a3Mat, a4Mat, a5Mat, rMat)
}

object TF5 {

  def apply[A1, A2, A3, A4, A5, R](func: (A1, A2, A3, A4, A5) => R): TF5[A1, A2, A3, A4, A5, R] =
    new TF5Cached[A1, A2, A3, A4, A5, R](func)

  class TF5Cached[A1, A2, A3, A4, A5, R](func: (A1, A2, A3, A4, A5) => R)
      extends TF5[A1, A2, A3, A4, A5, R] {

    private val cache = TF.Cache[R]()

    override def compileWith(session: Session)(
        implicit a1Mat: Mat[A1],
        a2Mat: Mat[A2],
        a3Mat: Mat[A3],
        a4Mat: Mat[A4],
        a5Mat: Mat[A5],
        rMat: Mat[R]): (a1Mat.Out, a2Mat.Out, a3Mat.Out, a4Mat.Out, a5Mat.Out) => rMat.Out = {
      (a1Out: a1Mat.Out, a2Out: a2Mat.Out, a3Out: a3Mat.Out, a4Out: a4Mat.Out, a5Out: a5Mat.Out) =>
        {
          val (a1Layout, a1T) = a1Mat.deconstructOut(a1Out)
          val (a2Layout, a2T) = a2Mat.deconstructOut(a2Out)
          val (a3Layout, a3T) = a3Mat.deconstructOut(a3Out)
          val (a4Layout, a4T) = a4Mat.deconstructOut(a4Out)
          val (a5Layout, a5T) = a5Mat.deconstructOut(a5Out)
          val aTAll = a1T ++ a2T ++ a3T ++ a4T ++ a5T
          val a1Type = a1T.map(tensor => (tensor.`type`, tensor.shape))
          val a2Type = a2T.map(tensor => (tensor.`type`, tensor.shape))
          val a3Type = a3T.map(tensor => (tensor.`type`, tensor.shape))
          val a4Type = a4T.map(tensor => (tensor.`type`, tensor.shape))
          val a5Type = a5T.map(tensor => (tensor.`type`, tensor.shape))
          val aShapes = a1Type.map(_._2) ++ a2Type.map(_._2) ++ a3Type.map(_._2) ++ a4Type.map(
            _._2) ++ a5Type.map(_._2)

          val (pAll, r) = cache.getOrCompute(aShapes) {
            val p1 = a1Type.map { case (t, s) => placeholderRaw(t, s) }
            val p2 = a2Type.map { case (t, s) => placeholderRaw(t, s) }
            val p3 = a3Type.map { case (t, s) => placeholderRaw(t, s) }
            val p4 = a4Type.map { case (t, s) => placeholderRaw(t, s) }
            val p5 = a5Type.map { case (t, s) => placeholderRaw(t, s) }
            val a1 = a1Mat.constructIn(a1Layout, p1)
            val a2 = a2Mat.constructIn(a2Layout, p2)
            val a3 = a3Mat.constructIn(a3Layout, p3)
            val a4 = a4Mat.constructIn(a4Layout, p4)
            val a5 = a5Mat.constructIn(a5Layout, p5)
            val r = func(a1, a2, a3, a4, a5)
            (p1 ++ p2 ++ p3 ++ p4 ++ p5, r)
          }

          val (rLayout, rIn) = rMat.deconstructIn(r)
          val rRaw = session.runner.feed(pAll zip aTAll: _*).evalUnsafe(rIn)
          rMat.constructOutRaw(rLayout, rRaw)
        }
    }
  }
}
