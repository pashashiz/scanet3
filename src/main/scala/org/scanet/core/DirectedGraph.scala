package org.scanet.core

import org.scanet.research.Expr

import scala.collection.mutable

/**
  * Mutable directed graph
  */
case class DirectedGraph[A](nodes: mutable.Map[String, Node[A]]) {

  def :+(node: Node[A]): DirectedGraph[A] = {
    if (!nodes.contains(node.id)) {
      nodes(node.id) = node
    }
    this
  }

  def link(from: String, to: String): DirectedGraph[A] = {
    val fromNode = nodes(from)
    val toNode = nodes(to)
    fromNode.output(toNode)
    toNode.input(fromNode)
    this
  }

  def linkAll(edges: Seq[(String, String)]): DirectedGraph[A] = {
    edges.foreach {case (from, to) => link(from, to)}
    this
  }

  def find(id: String): Option[Node[A]] = nodes.get(id)

  def apply(id: String): Node[A] = nodes(id)
}

object DirectedGraph {
  def apply[A](): DirectedGraph[A] = new DirectedGraph(mutable.Map())
}

case class Node[A](id: String, value: A, inputs: mutable.Buffer[Node[A]], outputs: mutable.Buffer[Node[A]]) {

  def isLeaf: Boolean = inputs.isEmpty
  def isRoot: Boolean = outputs.isEmpty

  def input(node: Node[A]): Node[A] = {
    inputs += node
    this
  }
  def output(node: Node[A]): Node[A] = {
    outputs += node
    this
  }

  override def hashCode(): Int = id.hashCode
  override def equals(obj: Any): Boolean = obj match {
    case other: Node[Expr] => other.id == id
    case _ => false
  }

  override def toString: String = s"Node(id=$id, value=$value, in=${inputs.map(_.id)}, out=${outputs.map(_.id)})"
}

object Node {
  def apply[A](id: String, value: A): Node[A] = new Node(id, value, mutable.Buffer(), mutable.Buffer())
}
