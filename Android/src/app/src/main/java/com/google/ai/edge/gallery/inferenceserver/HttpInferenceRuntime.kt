package com.google.ai.edge.gallery.inferenceserver

import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Process-wide state for the HTTP inference foreground service. */
@Singleton
class HttpInferenceRuntime @Inject constructor() {

  sealed class Status {
    data object Stopped : Status()

    data class Running(
      val modelName: String,
      val port: Int,
      /** False when the LLM engine was idle-unloaded but the HTTP server is still up. */
      val modelLoaded: Boolean = true,
    ) : Status()
  }

  private val _status = MutableStateFlow<Status>(Status.Stopped)
  val status: StateFlow<Status> = _status.asStateFlow()

  fun setRunning(modelName: String, port: Int, modelLoaded: Boolean = true) {
    _status.value = Status.Running(modelName = modelName, port = port, modelLoaded = modelLoaded)
  }

  fun setStopped() {
    _status.value = Status.Stopped
  }

  fun isBlockingModel(modelName: String): Boolean {
    val s = _status.value
    return s is Status.Running && s.modelName == modelName
  }
}
