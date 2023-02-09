package scanet

import org.tensorflow.internal.c_api.{AbstractTF_Status, TF_Status}

package object native {
  def attempt[A](f: TF_Status => A): A = {
    val status = AbstractTF_Status.newStatus()
    try f(status)
    finally status.throwExceptionIfNotOK()
  }
}
