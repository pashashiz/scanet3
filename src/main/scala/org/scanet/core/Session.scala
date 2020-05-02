package org.scanet.core

import org.scanet.core.Output.Context
import org.tensorflow.op.Scope
import org.tensorflow.{Graph, Operation, Output => NativeOutput, Session => NativeSession, Tensor => NativeTensor}

import scala.jdk.CollectionConverters._
import scala.{specialized => sp}
import scala.language.existentials

object Session {

  def run[@sp A1: TensorType](op: Output[A1]): Tensor[A1] = {
    val tensors = runN(List(op))
    Tensor[A1](tensors.head.asInstanceOf[NativeTensor[A1]])
  }

  def runN(ops: List[Output[_]]): Seq[NativeTensor[_]] = {
    val (graph, outputs) = compileN(ops)
    val session = new NativeSession(graph)
    try {
      val runner = outputs.reverse.foldLeft(session.runner)((runner, output) => runner.fetch(output))
      runner.run().asScala.toList
    } finally if (session != null) session.close()
  }

  def compileN(ops: List[Output[_]]): (Graph, Seq[NativeOutput[_]]) = {
    val graph = new Graph()
    val scope = new Scope(graph)
    val zero = (Context(scope, Map.empty), List[NativeOutput[_]]())
    val (_, outputs) = ops.foldLeft(zero)((acc, op) => {
      val (currentContext, outs) = acc
      val (nextContext: Context, (_, out: Operation)) = op.compile(currentContext)
      (nextContext, out.output(0)::outs)
    })
    (graph, outputs)
  }
}
