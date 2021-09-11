package scanet.native

import com.google.common.util.concurrent.ThreadFactoryBuilder

import java.util.concurrent.Executors
import scala.collection._
import scala.concurrent.{ExecutionContext, ExecutionContextExecutor, Future}
import scala.ref.{PhantomReference, Reference, ReferenceQueue}
import scala.util.control.NonFatal

/** Allows an object to register a deallocator
  * @param deallocator a deallocator which will be called after the object is garbage collected
  */
abstract class Disposable(deallocator: () => Unit) {
  private val ref = new PhantomReference[Disposable](this, Disposable.refs)
  Disposable.dealocators += (ref -> deallocator)
}

object Disposable {
  val dealocators: concurrent.Map[Reference[Disposable], () => Unit] = concurrent.TrieMap()
  val refs = new ReferenceQueue[Disposable]
  implicit val executor: ExecutionContextExecutor =
    ExecutionContext.fromExecutor(
      Executors.newSingleThreadExecutor(
        new ThreadFactoryBuilder().setNameFormat(s"scanet-deallocator-%d").build()))
  Future {
    while (true) {
      refs.remove.foreach(ref => {
        // that is a blocking call so we do not busy wait here
        dealocators
          .remove(ref)
          .foreach(deallocator => {
            try {
              deallocator()
            } catch {
              case _: InterruptedException => Thread.currentThread.interrupt()
              case NonFatal(e) =>
                Console.err.println("Error happened when cleaning up an object")
                e.printStackTrace()
            }
          })
      })
    }
  }
}
