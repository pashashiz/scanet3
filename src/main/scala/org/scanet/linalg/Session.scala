package org.scanet.linalg

import org.scanet.core.Numeric
import org.scanet.linalg.Op.Context
import org.tensorflow.op.Scope
import org.tensorflow.{Graph, Output, Session => NativeSession, Tensor => NativeTensor}

import collection.JavaConverters._
import scala.{specialized => sp}

object Session {

  def run[@sp A1: Numeric](op: Op[A1]): Tensor[A1] = {
    val tensors = runN(List(op))
    Tensor[A1](tensors.head.asInstanceOf[NativeTensor[A1]])
  }

  def runN(ops: List[Op[_]]): Seq[NativeTensor[_]] = {
    val graph = new Graph()
    val scope = new Scope(graph)
    val zero = (Context(scope, Map.empty), List[Output[_]]())
    val (_, outputs) = ops.foldLeft(zero)((acc, op) => {
      val (currentContext, outs) = acc
      val (nextContext: Context, (_, out)) = op.compile(currentContext)
      (nextContext, out::outs)
    })
    val session = new NativeSession(graph)
    try {
      val runner = outputs.reverse.foldLeft(session.runner)((runner, output) => runner.fetch(output))
      runner.run().asScala
    } finally if (session != null) session.close()
  }
}
