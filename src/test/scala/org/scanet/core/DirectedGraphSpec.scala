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
    graph("a").outputs.map(_.to) should contain (Node("b", "2"))
    graph("b").inputs.map(_.from) should contain (Node("a", "1"))
  }

  it should "support repeatable edges" in {
    val graph = DirectedGraph[String]() :+ Node("a", "1") :+ Node("b", "2") link("a", "b") link("a", "b")
    graph("a").outputs.map(_.to).count(_.id == "b") should be(2)
    graph("b").inputs.map(_.from).count(_.id == "a") should be(2)
  }

  it should "support self links" in {
    val graph = DirectedGraph[String]() :+ Node("a", "1") link("a", "a")
    graph("a").outputs.map(_.to) should contain (Node("a", "1"))
    graph("a").inputs.map(_.from) should contain (Node("a", "1"))
  }
}
