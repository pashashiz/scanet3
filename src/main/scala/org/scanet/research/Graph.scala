package org.scanet.research

import scala.collection.mutable

case class Graph[A](nodes: mutable.Map[String, Node[A]]) {

  def addNode(node: Node[A]): Graph[A] = {
    if (!nodes.contains(node.id)) {
      nodes(node.id) = node
    }
    this
  }

  def addNodes(all: Seq[Node[A]]): Graph[A] = {
    all.foreach(addNode)
    this
  }

  def addEdge(from: String, to: String): Graph[A] = {
    val fromNode = nodes(from)
    val toNode = nodes(to)
    fromNode.addOut(toNode)
    toNode.addIn(fromNode)
    this
  }

  def addEdges(edges: Seq[(String, String)]): Graph[A] = {
    edges.foreach {case (from, to) => addEdge(from, to)}
    this
  }

  def find(id: String): Option[Node[A]] = nodes.get(id)
}

object Graph {
  def apply[A](): Graph[A] = new Graph(mutable.Map())
}

case class Node[A](id: String, value: A, in: mutable.Buffer[Node[A]], out: mutable.Buffer[Node[A]]) {
  def isLeaf: Boolean = in.isEmpty
  def isRoot: Boolean = out.isEmpty
  def addIn(node: Node[A]): Node[A] = {
    in += node
    this
  }
  def addOut(node: Node[A]): Node[A] = {
    out += node
    this
  }

  override def hashCode(): Int = id.hashCode
  override def equals(obj: Any): Boolean = obj match {
    case other: Node[Expr] => other.id == id
    case _ => false
  }
  override def toString: String = s"Node(id=$id, value=$value, in=${in.map(_.id)}, out=${out.map(_.id)})"
}

object Node {
  def apply[A](id: String, value: A): Node[A] = new Node(id, value, mutable.Buffer(), mutable.Buffer())
}
