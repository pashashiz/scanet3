package org.scanet.linalg

import org.scanet.core.Numeric
import org.scanet.linalg.Op.Context
import org.tensorflow.op.Scope
import org.tensorflow.{Graph, Session => NativeSession, Tensor => NativeTensor}

import scala.{specialized => sp}

object Session {

  // (a + b).eval
  // (a, b).eval
  def run[@sp A1: Numeric](op: Op[A1]): Tensor[A1] = {
    val graph = new Graph()
    val scope = new Scope(graph)
    val (_, output) = op.compile(Context(scope, Map.empty))
    val s = new NativeSession(graph)
    try {
      val results = s.runner.fetch(output._2).run
      Tensor(results.get(0).asInstanceOf[NativeTensor[A1]])
    } finally if (s != null) s.close()

  }
}
