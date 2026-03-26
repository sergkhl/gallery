package com.google.ai.edge.gallery.inferenceserver

import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

private const val MAX_CHARS = 10_000

/** In-memory assistant output from HTTP /v1/chat/completions for on-device debugging. */
@Singleton
class HttpInferenceLlmDebugLog @Inject constructor() {

  private val lock = Any()
  private val buffer = StringBuilder()
  private val timeFormatter =
    DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS").withZone(ZoneId.systemDefault())

  private val _text = MutableStateFlow("")
  val text: StateFlow<String> = _text.asStateFlow()

  fun clear() {
    synchronized(lock) {
      buffer.clear()
      _text.value = ""
    }
  }

  /** Call once per chat completion request, before inference starts. */
  fun markCompletionStart() {
    val header = "---\n${timeFormatter.format(Instant.now())}\n"
    synchronized(lock) {
      appendAndTrim(header)
    }
  }

  fun appendDelta(delta: String) {
    if (delta.isEmpty()) return
    synchronized(lock) {
      appendAndTrim(delta)
    }
  }

  fun appendError(message: String) {
    if (message.isEmpty()) return
    synchronized(lock) {
      appendAndTrim("\nError: $message\n")
    }
  }

  private fun appendAndTrim(s: String) {
    buffer.append(s)
    if (buffer.length > MAX_CHARS) {
      val overflow = buffer.length - MAX_CHARS
      val nl = buffer.indexOf('\n', overflow)
      val endExclusive =
        if (nl >= 0) nl + 1 else overflow.coerceAtMost(buffer.length)
      buffer.delete(0, endExclusive.coerceAtMost(buffer.length))
    }
    _text.value = buffer.toString()
  }
}
