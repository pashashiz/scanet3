# MacOS ARM64 (M1)

At the moment Tensorflow Java do not publish the M1 compatible binary (see [this](https://github.com/tensorflow/java/blob/master/CONTRIBUTING.md#apple-silicon))
cause GitHub Actions still has no such build workers. For now, we have to build it locally from source:
 - Checkout [Tensorflow Java](https://github.com/tensorflow/java)
 - Install env for Tensorflow, see [Build from source](https://www.tensorflow.org/install/source#macos). 
   **IMPORTANT**t As of `01-01-2023`, TensorFlow fails to build on XCode command line tools version `14+`. If you have such version installed, it might be necessary to downgrade it to a [previous](https://developer.apple.com/download/all/?q=xcode) version, like `13.4.1`.
 - Run `mvn install` (requires JDK 11)

We will be tracking [Strategies for getting Tensorflow-Java on Apple Silicon?
#394](https://github.com/tensorflow/java/issues/394) to remove that chore.