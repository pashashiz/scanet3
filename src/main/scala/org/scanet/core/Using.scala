package org.scanet.core

import scala.util.{Success, Failure, Try}

object Using {
  def resource[A <: AutoCloseable, B](resource: A)(block: A => B): B = {
    Try(block(resource)) match {
      case Success(result) =>
        resource.close()
        result
      case Failure(e) =>
        resource.close()
        throw e
    }
  }
}
