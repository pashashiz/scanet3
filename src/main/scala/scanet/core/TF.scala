package scanet.core

import scanet.math.syntax.placeholder
import scala.collection.immutable.Seq

trait TF1[P1, P1M, R] {

  private val buildMemoized = memoize(s => build(s))

  def materializedToSeq(in: P1M): Seq[Tensor[P1]]

  def build(shapes: Seq[Shape]): (Seq[Expr[P1]], R)

  def compile()(implicit res: CanEval[R]): P1M => res.Materialized = compile(new Session())(res)

  def compile(session: Session)(implicit res: CanEval[R]): P1M => res.Materialized = { in: P1M =>
    {
      val t1: Seq[Tensor[P1]] = materializedToSeq(in)
      val (p1, out) = buildMemoized(t1.map(_.shape))
      session.runner.feed(p1 zip t1: _*).eval(out)(res)
    }
  }

  def combine[P1Other, P1MOther, ROther, RNew](other: TF1[P1Other, P1MOther, ROther])(
      via: (R, ROther) => RNew): TF2[P1, P1M, P1Other, P1MOther, RNew] =
    new TF2[P1, P1M, P1Other, P1MOther, RNew] {

      override def materializedToSeq(in1: P1M, in2: P1MOther) =
        (TF1.this.materializedToSeq(in1), other.materializedToSeq(in2))

      override def build(
          s1: Seq[Shape],
          s2: Seq[Shape]): (Seq[Expr[P1]], Seq[Expr[P1Other]], RNew) = {
        val (p1, out) = TF1.this.build(s1)
        val (p1Other, outOther) = other.build(s2)
        (p1, p1Other, via(out, outOther))
      }
    }

  def display(s1: Seq[Shape], label: String = "", dir: String = "")(
      implicit res: CanEval[R]): Unit = {
    val (_, r) = build(s1)
    val outs = res.unwrap(r).map(out => if (label.nonEmpty) out.as(label) else out)
    TensorBoard(dir).addGraph(outs: _*)
  }
}

// format: off
object TF1 {

  def apply[A1[_], P1: TensorType, R](func: A1[P1] => R)(implicit arg1: OutputContainer[A1]): TF1[P1, arg1.Materialized[P1], R] =
    new TF1[P1, arg1.Materialized[P1], R] {

      override def materializedToSeq(in: arg1.Materialized[P1]): Seq[Tensor[P1]] = arg1.materializedToSeq(in)

      override def build(shapes: Seq[Shape]): (Seq[Expr[P1]], R) = {
        val p1 = shapes.map(placeholder[P1](_))
        val out = func(arg1.of(p1))
        (p1, out)
      }
    }

  def identity[A1[_], P1: TensorType](implicit arg1: OutputContainer[A1]) =
    TF1[A1, P1, A1[P1]](arg => arg)
}

trait TF2[P1, P1M, P2, P2M, R] {

  private val buildMemoized = memoize((s1, s2) => build(s1, s2))

  def materializedToSeq(in1: P1M, in2: P2M): (Seq[Tensor[P1]], Seq[Tensor[P2]])

  def build(s1: Seq[Shape], s2: Seq[Shape]): (Seq[Expr[P1]], Seq[Expr[P2]], R)

  def compile()(implicit res: CanEval[R]): (P1M, P2M) => res.Materialized = compile(new Session())(res)

  def compile(session: Session)(implicit res: CanEval[R]): (P1M, P2M) => res.Materialized = {
    (in1: P1M, in2: P2M) => {
      val (t1, t2) = materializedToSeq(in1, in2)
      val (p1, p2, out) = buildMemoized(t1.map(_.shape), t2.map(_.shape))
      session.runner.feed(p1 zip t1: _*).feed(p2 zip t2: _*).eval(out)(res)
    }
  }

  def display(s1: Seq[Shape], s2: Seq[Shape], label: String = "", dir: String = "")(implicit res: CanEval[R]): Unit = {
    val (_, _, r) = build(s1, s2)
    val outs = res.unwrap(r)
      .map(out => if (label.nonEmpty) out.as(label) else out)
    TensorBoard(dir).addGraph(outs: _*)
  }
}

object TF2 {

  def apply[A1[_], P1: TensorType, A2[_], P2: TensorType, R](func: (A1[P1], A2[P2]) => R)(implicit arg1: OutputContainer[A1], arg2: OutputContainer[A2]): TF2[P1, arg1.Materialized[P1], P2, arg2.Materialized[P2], R] =
    new TF2[P1, arg1.Materialized[P1], P2, arg2.Materialized[P2], R] {

      override def materializedToSeq(in1: arg1.Materialized[P1], in2: arg2.Materialized[P2]) =
        (arg1.materializedToSeq(in1), arg2.materializedToSeq(in2))

      override def build(s1: Seq[Shape], s2: Seq[Shape]): (Seq[Expr[P1]], Seq[Expr[P2]], R) = {
        val p1 = s1.map(placeholder[P1](_))
        val p2 = s2.map(placeholder[P2](_))
        val out = func(arg1.of(p1), arg2.of(p2))
        (p1, p2, out)
      }
    }

  def identity[A1[_], P1: TensorType, A2[_], P2: TensorType](implicit arg1: OutputContainer[A1], arg2: OutputContainer[A2]) =
    TF2[A1, P1, A2, P2, (A1[P1], A2[P2])]((arg1, arg2) => (arg1, arg2))
}

trait TF3[P1, P1M, P2, P2M, P3, P3M, R] {

  private val buildMemoized = memoize((s1, s2, s3) => build(s1, s2, s3))

  def materializedToSeq(in1: P1M, in2: P2M, in3: P3M): (Seq[Tensor[P1]], Seq[Tensor[P2]], Seq[Tensor[P3]])

  def build(s1: Seq[Shape], s2: Seq[Shape], s3: Seq[Shape]): (Seq[Expr[P1]], Seq[Expr[P2]], Seq[Expr[P3]], R)

  def compile()(implicit res: CanEval[R]): (P1M, P2M, P3M) => res.Materialized = compile(new Session())(res)

  def compile(session: Session)(implicit res: CanEval[R]): (P1M, P2M, P3M) => res.Materialized = {
    (in1: P1M, in2: P2M, in3: P3M) => {
      val (t1, t2, t3) = materializedToSeq(in1, in2, in3)
      val (p1, p2, p3, out) = buildMemoized(t1.map(_.shape), t2.map(_.shape), t3.map(_.shape))
      session.runner.feed(p1 zip t1: _*).feed(p2 zip t2: _*).feed(p3 zip t3: _*).eval(out)(res)
    }
  }

  def combine[P1Other, P1MOther, P2Other, P2MOther, ROther, RNew](other: TF2[P1Other, P1MOther, P2Other, P2MOther, ROther])(via: (R, ROther) => RNew): TF5[P1, P1M, P2, P2M, P3, P3M, P1Other, P1MOther, P2Other, P2MOther, RNew] = new TF5[P1, P1M, P2, P2M, P3, P3M, P1Other, P1MOther, P2Other, P2MOther, RNew] {

    override def materializedToSeq(in1: P1M, in2: P2M, in3: P3M, in4: P1MOther, in5: P2MOther): (Seq[Tensor[P1]], Seq[Tensor[P2]], Seq[Tensor[P3]], Seq[Tensor[P1Other]], Seq[Tensor[P2Other]]) = {
      val (t1, t2, t3) = TF3.this.materializedToSeq(in1, in2, in3)
      val (t4, t5) = other.materializedToSeq(in4, in5)
      (t1, t2, t3, t4, t5)
    }

    override def build(s1: Seq[Shape], s2: Seq[Shape], s3: Seq[Shape], s4: Seq[Shape], s5: Seq[Shape]): (Seq[Expr[P1]], Seq[Expr[P2]], Seq[Expr[P3]], Seq[Expr[P1Other]], Seq[Expr[P2Other]], RNew) = {
      val (p1, p2, p3, out) = TF3.this.build(s1, s2, s3)
      val (p1Other, p2Other, outOther) = other.build(s4, s5)
      (p1, p2, p3, p1Other, p2Other, via(out, outOther))
    }
  }
  def display(s1: Seq[Shape], s2: Seq[Shape], s3: Seq[Shape], label: String = "", dir: String = "")(implicit res: CanEval[R]): Unit = {
    val (_, _, _, r) = build(s1, s2, s3)
    val outs = res.unwrap(r)
      .map(out => if (label.nonEmpty) out.as(label) else out)
    TensorBoard(dir).addGraph(outs: _*)
  }
}

object TF3 {

  def apply[A1[_], P1: TensorType, A2[_], P2: TensorType, A3[_], P3: TensorType, R](func: (A1[P1], A2[P2], A3[P3]) => R)(implicit arg1: OutputContainer[A1], arg2: OutputContainer[A2], arg3: OutputContainer[A3]): TF3[P1, arg1.Materialized[P1], P2, arg2.Materialized[P2], P3, arg3.Materialized[P3], R] =
    new TF3[P1, arg1.Materialized[P1], P2, arg2.Materialized[P2], P3, arg3.Materialized[P3], R] {

      override def materializedToSeq(in1: arg1.Materialized[P1], in2: arg2.Materialized[P2], in3: arg3.Materialized[P3]) =
        (arg1.materializedToSeq(in1), arg2.materializedToSeq(in2), arg3.materializedToSeq(in3))

      override def build(s1: Seq[Shape], s2: Seq[Shape], s3: Seq[Shape]): (Seq[Expr[P1]], Seq[Expr[P2]], Seq[Expr[P3]], R) = {
        val p1 = s1.map(placeholder[P1](_))
        val p2 = s2.map(placeholder[P2](_))
        val p3 = s3.map(placeholder[P3](_))
        val out = func(arg1.of(p1), arg2.of(p2), arg3.of(p3))
        (p1, p2, p3, out)
      }
    }
}

trait TF4[P1, P1M, P2, P2M, P3, P3M, P4, P4M, R] {

  private val buildMemoized = memoize((s1, s2, s3, s4) => build(s1, s2, s3, s4))

  def materializedToSeq(in1: P1M, in2: P2M, in3: P3M, in4: P4M): (Seq[Tensor[P1]], Seq[Tensor[P2]], Seq[Tensor[P3]], Seq[Tensor[P4]])

  def build(s1: Seq[Shape], s2: Seq[Shape], s3: Seq[Shape], s4: Seq[Shape]): (Seq[Expr[P1]], Seq[Expr[P2]], Seq[Expr[P3]], Seq[Expr[P4]], R)

  def compile()(implicit res: CanEval[R]): (P1M, P2M, P3M, P4M) => res.Materialized = compile(new Session())(res)

  def compile(session: Session)(implicit res: CanEval[R]): (P1M, P2M, P3M, P4M) => res.Materialized = {
    (in1: P1M, in2: P2M, in3: P3M, in4: P4M) => {
      val (t1, t2, t3, t4) = materializedToSeq(in1, in2, in3, in4)
      val (p1, p2, p3, p4, out) = buildMemoized(t1.map(_.shape), t2.map(_.shape), t3.map(_.shape), t4.map(_.shape))
      session.runner.feed(p1 zip t1: _*).feed(p2 zip t2: _*).feed(p3 zip t3: _*).feed(p4 zip t4: _*).eval(out)(res)
    }
  }

  def display(s1: Seq[Shape], s2: Seq[Shape], s3: Seq[Shape], s4: Seq[Shape], label: String = "", dir: String = "")(implicit res: CanEval[R]): Unit = {
    val (_, _, _, _, r) = build(s1, s2, s3, s4)
    val outs = res.unwrap(r)
      .map(out => if (label.nonEmpty) out.as(label) else out)
    TensorBoard(dir).addGraph(outs: _*)
  }
}

object TF4 {

  def apply[A1[_], P1: TensorType, A2[_], P2: TensorType, A3[_], P3: TensorType, A4[_], P4: TensorType, R](func: (A1[P1], A2[P2], A3[P3], A4[P4]) => R)(implicit arg1: OutputContainer[A1], arg2: OutputContainer[A2], arg3: OutputContainer[A3], arg4: OutputContainer[A4]): TF4[P1, arg1.Materialized[P1], P2, arg2.Materialized[P2], P3, arg3.Materialized[P3], P4, arg4.Materialized[P4], R] =
    new TF4[P1, arg1.Materialized[P1], P2, arg2.Materialized[P2], P3, arg3.Materialized[P3], P4, arg4.Materialized[P4], R] {

      override def materializedToSeq(in1: arg1.Materialized[P1], in2: arg2.Materialized[P2], in3: arg3.Materialized[P3], in4: arg4.Materialized[P4]) =
        (arg1.materializedToSeq(in1), arg2.materializedToSeq(in2), arg3.materializedToSeq(in3), arg4.materializedToSeq(in4))

      override def build(s1: Seq[Shape], s2: Seq[Shape], s3: Seq[Shape], s4: Seq[Shape]): (Seq[Expr[P1]], Seq[Expr[P2]], Seq[Expr[P3]], Seq[Expr[P4]], R) = {
        val p1 = s1.map(placeholder[P1](_))
        val p2 = s2.map(placeholder[P2](_))
        val p3 = s3.map(placeholder[P3](_))
        val p4 = s3.map(placeholder[P4](_))
        val out = func(arg1.of(p1), arg2.of(p2), arg3.of(p3), arg4.of(p4))
        (p1, p2, p3, p4, out)
      }
    }
}

trait TF5[P1, P1M, P2, P2M, P3, P3M, P4, P4M, P5, P5M, R] {

  private val buildMemoized = memoize((s1, s2, s3, s4, s5) => build(s1, s2, s3, s4, s5))

  def materializedToSeq(in1: P1M, in2: P2M, in3: P3M, in4: P4M, in5: P5M): (Seq[Tensor[P1]], Seq[Tensor[P2]], Seq[Tensor[P3]], Seq[Tensor[P4]], Seq[Tensor[P5]])

  def build(s1: Seq[Shape], s2: Seq[Shape], s3: Seq[Shape], s4: Seq[Shape], s5: Seq[Shape]): (Seq[Expr[P1]], Seq[Expr[P2]], Seq[Expr[P3]], Seq[Expr[P4]], Seq[Expr[P5]], R)

  def compile()(implicit res: CanEval[R]): (P1M, P2M, P3M, P4M, P5M) => res.Materialized = compile(new Session())(res)

  def compile(session: Session)(implicit res: CanEval[R]): (P1M, P2M, P3M, P4M, P5M) => res.Materialized = {
    (in1: P1M, in2: P2M, in3: P3M, in4: P4M, in5: P5M) => {
      val (t1, t2, t3, t4, t5) = materializedToSeq(in1, in2, in3, in4, in5)
      val (p1, p2, p3, p4, p5, out) = buildMemoized(t1.map(_.shape), t2.map(_.shape), t3.map(_.shape), t4.map(_.shape), t5.map(_.shape))
      session.runner.feed(p1 zip t1: _*).feed(p2 zip t2: _*).feed(p3 zip t3: _*).feed(p4 zip t4: _*).feed(p5 zip t5: _*).eval(out)(res)
    }
  }

  def display(s1: Seq[Shape], s2: Seq[Shape], s3: Seq[Shape], s4: Seq[Shape], s5: Seq[Shape], label: String = "", dir: String = "")(implicit res: CanEval[R]): Unit = {
    val (_, _, _, _, _, r) = build(s1, s2, s3, s4, s5)
    val outs = res.unwrap(r)
      .map(out => if (label.nonEmpty) out.as(label) else out)
    TensorBoard(dir).addGraph(outs: _*)
  }
}

object TF5 {

  def apply[A1[_], P1: TensorType, A2[_], P2: TensorType, A3[_], P3: TensorType, A4[_], P4: TensorType, A5[_], P5: TensorType, R](func: (A1[P1], A2[P2], A3[P3], A4[P4], A5[P5]) => R)(implicit arg1: OutputContainer[A1], arg2: OutputContainer[A2], arg3: OutputContainer[A3], arg4: OutputContainer[A4], arg5: OutputContainer[A5]): TF5[P1, arg1.Materialized[P1], P2, arg2.Materialized[P2], P3, arg3.Materialized[P3], P4, arg4.Materialized[P4], P5, arg5.Materialized[P5], R] =
    new TF5[P1, arg1.Materialized[P1], P2, arg2.Materialized[P2], P3, arg3.Materialized[P3], P4, arg4.Materialized[P4], P5, arg5.Materialized[P5], R] {

      override def materializedToSeq(in1: arg1.Materialized[P1], in2: arg2.Materialized[P2], in3: arg3.Materialized[P3], in4: arg4.Materialized[P4], in5: arg5.Materialized[P5]) =
        (arg1.materializedToSeq(in1), arg2.materializedToSeq(in2), arg3.materializedToSeq(in3), arg4.materializedToSeq(in4), arg5.materializedToSeq(in5))

      override def build(s1: Seq[Shape], s2: Seq[Shape], s3: Seq[Shape], s4: Seq[Shape], s5: Seq[Shape]): (Seq[Expr[P1]], Seq[Expr[P2]], Seq[Expr[P3]], Seq[Expr[P4]], Seq[Expr[P5]], R) = {
        val p1 = s1.map(placeholder[P1](_))
        val p2 = s2.map(placeholder[P2](_))
        val p3 = s3.map(placeholder[P3](_))
        val p4 = s3.map(placeholder[P4](_))
        val p5 = s3.map(placeholder[P5](_))
        val out = func(arg1.of(p1), arg2.of(p2), arg3.of(p3), arg4.of(p4), arg5.of(p5))
        (p1, p2, p3, p4, p5, out)
      }
    }
}
// format: on
