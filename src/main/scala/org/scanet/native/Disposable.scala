package org.scanet.native

import java.util.concurrent.Executors

import scala.concurrent.{ExecutionContext, ExecutionContextExecutor, Future}
import scala.ref.{PhantomReference, ReferenceQueue}
import scala.util.control.NonFatal

/**
 * Allows an object to register a deallocator
 * @param deallocator a deallocator which will be called after the object is garbage collected
 */
abstract class Disposable(private[Disposable] val deallocator: () => Unit) {
  new PhantomReference(this, Disposable.refs)
}

object Disposable {
  val refs = new ReferenceQueue[Disposable]
  implicit val executor: ExecutionContextExecutor =
    ExecutionContext.fromExecutorService(Executors.newSingleThreadExecutor())
  Future {
    while (true) {
      // that is a blocking call so we do not busy wait here
      refs.remove.foreach(ref => {
        try {
          ref.get.foreach(_.deallocator())
        } catch {
          case _: InterruptedException => Thread.currentThread.interrupt()
          case NonFatal(e) =>
            Console.err.println("Error happened when cleaning up an object")
            e.printStackTrace()
        }
      })
    }
  }
}
