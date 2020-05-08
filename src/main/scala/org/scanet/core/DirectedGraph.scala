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
    val edge = Edge(toNode.inputs.size, fromNode, toNode)
    fromNode.output(edge)
    toNode.input(edge)
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

case class Node[A](id: String, value: A, inputs: mutable.Buffer[Edge[A]], outputs: mutable.Buffer[Edge[A]]) {

  def isLeaf: Boolean = inputs.isEmpty
  def isRoot: Boolean = outputs.isEmpty

  def input(edge: Edge[A]): Node[A] = {
    inputs += edge
    this
  }
  def output(edge: Edge[A]): Node[A] = {
    outputs += edge
    this
  }

  override def hashCode(): Int = id.hashCode
  override def equals(obj: Any): Boolean = obj match {
    case other: Node[_] => other.id == id
    case _ => false
  }

  override def toString: String = s"Node(id=$id, value=$value, in=${inputs.map(_.from.id)}, out=${outputs.map(_.to.id)})"
}

case class Edge[A](index: Int, from: Node[A], to: Node[A])

object Node {
  def apply[A](id: String, value: A): Node[A] = new Node(id, value, mutable.Buffer(), mutable.Buffer())
}
