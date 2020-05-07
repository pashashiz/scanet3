package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DirectedGraphSpec extends AnyFlatSpec with Matchers {

  "graph" should "add new nodes correctly" in {
    val graph = DirectedGraph[String]() :+ Node("a", "1") :+ Node("b", "2")
    graph.find("a") should be(Some(Node("a", "1")))
    graph.find("b") should be(Some(Node("b", "2")))
    graph.find("c") should be(None)
  }

  it should "link nodes correctly" in {
    val graph = DirectedGraph[String]() :+ Node("a", "1") :+ Node("b", "2") link("a", "b")
    graph("a").outputs should contain (Node("b", "2"))
    graph("b").inputs should contain (Node("a", "1"))
  }

  it should "support repeatable edges" in {
    val graph = DirectedGraph[String]() :+ Node("a", "1") :+ Node("b", "2") link("a", "b") link("a", "b")
    graph("a").outputs.count(_.id == "b") should be(2)
    graph("b").inputs.count(_.id == "a") should be(2)
  }

  it should "support self links" in {
    val graph = DirectedGraph[String]() :+ Node("a", "1") link("a", "a")
    graph("a").outputs should contain (Node("a", "1"))
    graph("a").inputs should contain (Node("a", "1"))
  }
}
