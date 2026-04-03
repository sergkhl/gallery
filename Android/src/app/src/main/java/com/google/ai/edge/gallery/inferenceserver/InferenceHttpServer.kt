package com.google.ai.edge.gallery.inferenceserver

import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.gallery.data.DataStoreRepository
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.inferenceserver.SecurityGateway.Decision
import com.google.ai.edge.gallery.inferenceserver.dto.ChatCompletionChunk
import com.google.ai.edge.gallery.inferenceserver.dto.ChatCompletionChunkChoice
import com.google.ai.edge.gallery.inferenceserver.dto.ChatCompletionChoice
import com.google.ai.edge.gallery.inferenceserver.dto.ChatCompletionDelta
import com.google.ai.edge.gallery.inferenceserver.dto.ChatCompletionMessage
import com.google.ai.edge.gallery.inferenceserver.dto.ChatCompletionRequest
import com.google.ai.edge.gallery.inferenceserver.dto.ChatCompletionResponse
import com.google.ai.edge.gallery.inferenceserver.dto.HttpInferenceInputParseResult
import com.google.ai.edge.gallery.inferenceserver.dto.ModelsListResponse
import com.google.ai.edge.gallery.inferenceserver.dto.OpenAiModelInfo
import com.google.ai.edge.gallery.inferenceserver.dto.messagesToHttpInferenceInput
import com.google.ai.edge.gallery.inferenceserver.dto.recycleHttpInferenceImages
import com.google.ai.edge.gallery.ui.llmchat.LlmChatModelHelper
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpMethod
import io.ktor.http.HttpStatusCode
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.ApplicationCall
import io.ktor.server.application.install
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.cors.routing.CORS
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.response.respondTextWriter
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import java.util.UUID
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.serialization.json.Json

private val openAiJson =
  Json {
    ignoreUnknownKeys = true
    encodeDefaults = true
  }

/** Max time to wait for the inference lock before responding HTTP 429 (server busy). */
private const val INFERENCE_LOCK_WAIT_MS = 2000L

private const val INFERENCE_LOCK_POLL_MS = 40L

private const val TAG = "AGInferenceHttp"

/** Max time to wait for LiteRT callbacks to finish after stop before releasing the inference lock. */
private const val INFERENCE_IDLE_WAIT_MS = 5000L

private sealed interface HttpInferenceStreamEvent {
  data class Token(val text: String) : HttpInferenceStreamEvent

  data class ThinkingToken(val text: String) : HttpInferenceStreamEvent

  data object End : HttpInferenceStreamEvent

  data class Error(val message: String) : HttpInferenceStreamEvent
}

fun Application.configureInferenceOpenAiRoutes(
  securityGateway: SecurityGateway,
  dataStoreRepository: DataStoreRepository,
  inferenceModelRegistry: InferenceModelRegistry,
  inferenceMutex: Mutex,
  serviceScope: CoroutineScope,
  boundModelName: String,
  ensureLlmLoaded: suspend () -> Model,
  onInferenceActivity: () -> Unit,
  llmDebugLog: HttpInferenceLlmDebugLog,
  enableThinkingDefault: Boolean = false,
) {
  install(ContentNegotiation) { json(openAiJson) }

  install(CORS) {
    anyHost()
    allowHeader(HttpHeaders.Authorization)
    allowHeader(HttpHeaders.ContentType)
    allowMethod(HttpMethod.Options)
    allowMethod(HttpMethod.Get)
    allowMethod(HttpMethod.Post)
  }

  routing {
    get("/v1/models") {
      val clientIp = call.request.local.remoteHost
      if (!verifyBearer(call, dataStoreRepository)) {
        call.respond(HttpStatusCode.Unauthorized)
        return@get
      }
      if (!passesSecurity(clientIp, securityGateway)) {
        call.respond(HttpStatusCode.Forbidden)
        return@get
      }
      onInferenceActivity()
      val models =
        inferenceModelRegistry.downloadedLlmModels.value.map {
          OpenAiModelInfo(id = it.name)
        }
      call.respond(ModelsListResponse(data = models))
    }

    post("/v1/chat/completions") {
      val clientIp = call.request.local.remoteHost
      if (!verifyBearer(call, dataStoreRepository)) {
        call.respond(HttpStatusCode.Unauthorized)
        return@post
      }
      if (!passesSecurity(clientIp, securityGateway)) {
        call.respond(HttpStatusCode.Forbidden)
        return@post
      }

      val req = call.receive<ChatCompletionRequest>()
      if (req.model.isNotBlank() && req.model != boundModelName) {
        call.respond(
          HttpStatusCode.BadRequest,
          mapOf(
            "error" to
              mapOf(
                "message" to
                  "This server is bound to model '$boundModelName'. Requested '${req.model}'."
              )
          ),
        )
        return@post
      }

      val lockDeadlineNs = System.nanoTime() + INFERENCE_LOCK_WAIT_MS * 1_000_000
      while (!inferenceMutex.tryLock()) {
        if (System.nanoTime() >= lockDeadlineNs) {
          call.respond(HttpStatusCode.TooManyRequests)
          return@post
        }
        delay(INFERENCE_LOCK_POLL_MS)
      }

      var inferenceIdle: CompletableDeferred<Unit>? = null
      var chatModel: Model? = null
      try {
        chatModel =
          try {
            ensureLlmLoaded()
          } catch (e: CancellationException) {
            throw e
          } catch (e: Exception) {
            Log.e(TAG, "Failed to load LLM for HTTP inference", e)
            call.respond(
              HttpStatusCode.ServiceUnavailable,
              mapOf(
                "error" to
                  mapOf(
                    "message" to
                      (e.message ?: "Model failed to load. Try again or check HTTP inference settings.")
                  )
              ),
            )
            return@post
          }
        val boundModel = chatModel!!
        val inferenceInput = messagesToHttpInferenceInput(req)
        val prompt: String
        val images: List<Bitmap>
        when (inferenceInput) {
          is HttpInferenceInputParseResult.Error -> {
            call.respond(
              HttpStatusCode.BadRequest,
              mapOf("error" to mapOf("message" to inferenceInput.message)),
            )
            return@post
          }
          is HttpInferenceInputParseResult.Ok -> {
            prompt = inferenceInput.prompt
            images = inferenceInput.images
          }
        }
        if (images.isNotEmpty() && !boundModel.llmSupportImage) {
          images.recycleHttpInferenceImages()
          call.respond(
            HttpStatusCode.BadRequest,
            mapOf(
              "error" to
                mapOf(
                  "message" to
                    "This model does not support images. Use a vision-capable model or send text-only content."
                )
            ),
          )
          return@post
        }
        llmDebugLog.markCompletionStart()
        val enableThinking =
          (req.enableThinking ?: enableThinkingDefault) && boundModel.llmSupportThinking
        val extraContext =
          if (enableThinking) mapOf("enable_thinking" to "true") else null
        if (req.stream) {
          val streamId = "chatcmpl-${UUID.randomUUID()}"
          val idle = CompletableDeferred<Unit>()
          inferenceIdle = idle
          call.respondTextWriter(contentType = ContentType.Text.EventStream) {
            writeStreamingChat(
              serviceScope = serviceScope,
              model = boundModel,
              prompt = prompt,
              images = images,
              streamId = streamId,
              modelName = boundModel.name,
              inferenceIdle = idle,
              llmDebugLog = llmDebugLog,
              extraContext = extraContext,
            ) { chunk ->
              write(chunk)
              flush()
            }
          }
        } else {
          val blockingResult =
            runBlockingChat(
              serviceScope = serviceScope,
              model = boundModel,
              prompt = prompt,
              images = images,
              llmDebugLog = llmDebugLog,
              extraContext = extraContext,
            )
          inferenceIdle = blockingResult.inferenceIdle
          call.respond(
            ChatCompletionResponse(
              id = "chatcmpl-${UUID.randomUUID()}",
              model = boundModel.name,
              choices =
                listOf(
                  ChatCompletionChoice(
                    message =
                      ChatCompletionMessage(
                        content = blockingResult.content,
                        reasoningContent = blockingResult.reasoningContent.ifEmpty { null },
                      )
                  )
                ),
            )
          )
        }
      } finally {
        withContext(NonCancellable) {
          chatModel?.let { LlmChatModelHelper.stopResponse(it) }
          val idle = inferenceIdle
          if (idle != null) {
            val finished =
              withTimeoutOrNull(INFERENCE_IDLE_WAIT_MS) {
                idle.await()
              }
            if (finished == null) {
              Log.w(
                TAG,
                "Timed out waiting for native inference idle (${INFERENCE_IDLE_WAIT_MS}ms) before unlocking",
              )
            }
          }
          onInferenceActivity()
          inferenceMutex.unlock()
        }
      }
    }
  }
}

private sealed interface StreamChunk {
  data class Content(val text: String) : StreamChunk
  data class Reasoning(val text: String) : StreamChunk
  data class StreamError(val message: String) : StreamChunk
}

private suspend fun writeStreamingChat(
  serviceScope: CoroutineScope,
  model: Model,
  prompt: String,
  images: List<Bitmap>,
  streamId: String,
  modelName: String,
  inferenceIdle: CompletableDeferred<Unit>,
  llmDebugLog: HttpInferenceLlmDebugLog,
  extraContext: Map<String, String>? = null,
  emit: suspend (String) -> Unit,
) {
  fun markInferenceIdle() {
    if (!inferenceIdle.isCompleted) {
      inferenceIdle.complete(Unit)
    }
  }

  val out = Channel<StreamChunk>(Channel.UNLIMITED)
  val mailbox = Channel<HttpInferenceStreamEvent>(Channel.UNLIMITED)
  serviceScope.launch {
    try {
      for (event in mailbox) {
        when (event) {
          is HttpInferenceStreamEvent.Token -> {
            if (event.text.isNotEmpty()) {
              llmDebugLog.appendDelta(event.text)
              out.send(StreamChunk.Content(event.text))
            }
          }
          is HttpInferenceStreamEvent.ThinkingToken -> {
            if (event.text.isNotEmpty()) {
              out.send(StreamChunk.Reasoning(event.text))
            }
          }
          is HttpInferenceStreamEvent.Error -> {
            llmDebugLog.appendError(event.message)
            out.send(StreamChunk.StreamError(event.message))
            return@launch
          }
          is HttpInferenceStreamEvent.End -> return@launch
        }
      }
    } finally {
      out.close()
      markInferenceIdle()
      mailbox.close()
    }
  }
  LlmChatModelHelper.resetConversation(
    model = model,
    supportImage = model.llmSupportImage,
    supportAudio = model.llmSupportAudio,
    systemInstruction = null,
    tools = emptyList(),
    enableConversationConstrainedDecoding = false,
  )
  LlmChatModelHelper.runInference(
    model = model,
    input = prompt,
    resultListener = { partial, done, partialThinking ->
      if (partial.startsWith("<ctrl")) {
        if (done) {
          mailbox.trySend(HttpInferenceStreamEvent.End)
        }
        return@runInference
      }
      if (!partialThinking.isNullOrEmpty()) {
        mailbox.trySend(HttpInferenceStreamEvent.ThinkingToken(partialThinking))
      }
      if (partial.isNotEmpty()) {
        mailbox.trySend(HttpInferenceStreamEvent.Token(partial))
      }
      if (done) {
        mailbox.trySend(HttpInferenceStreamEvent.End)
      }
    },
    cleanUpListener = {},
    onError = { message ->
      mailbox.trySend(HttpInferenceStreamEvent.Error(message))
    },
    images = images,
    coroutineScope = serviceScope,
    extraContext = extraContext,
  )
  for (chunk in out) {
    when (chunk) {
      is StreamChunk.StreamError -> {
        emit(
          "data: {\"error\":{\"message\":\"${chunk.message.replace("\"", "\\\"")}\"}}\n\n"
        )
        emit("data: [DONE]\n\n")
        return
      }
      is StreamChunk.Reasoning -> {
        val sse =
          ChatCompletionChunk(
            id = streamId,
            model = modelName,
            choices =
              listOf(
                ChatCompletionChunkChoice(
                  delta = ChatCompletionDelta(reasoningContent = chunk.text),
                  finishReason = null,
                )
              ),
          )
        emit("data: ${openAiJson.encodeToString(ChatCompletionChunk.serializer(), sse)}\n\n")
      }
      is StreamChunk.Content -> {
        val sse =
          ChatCompletionChunk(
            id = streamId,
            model = modelName,
            choices =
              listOf(
                ChatCompletionChunkChoice(
                  delta = ChatCompletionDelta(content = chunk.text),
                  finishReason = null,
                )
              ),
          )
        emit("data: ${openAiJson.encodeToString(ChatCompletionChunk.serializer(), sse)}\n\n")
      }
    }
  }
  val finalChunk =
    ChatCompletionChunk(
      id = streamId,
      model = modelName,
      choices =
        listOf(
          ChatCompletionChunkChoice(
            delta = ChatCompletionDelta(content = null),
            finishReason = "stop",
          )
        ),
    )
  emit("data: ${openAiJson.encodeToString(ChatCompletionChunk.serializer(), finalChunk)}\n\n")
  emit("data: [DONE]\n\n")
}

private data class BlockingChatResult(
  val content: String,
  val reasoningContent: String,
  val inferenceIdle: CompletableDeferred<Unit>,
)

private suspend fun runBlockingChat(
  serviceScope: CoroutineScope,
  model: Model,
  prompt: String,
  images: List<Bitmap>,
  llmDebugLog: HttpInferenceLlmDebugLog,
  extraContext: Map<String, String>? = null,
): BlockingChatResult {
  val inferenceIdle = CompletableDeferred<Unit>()
  fun markInferenceIdle() {
    if (!inferenceIdle.isCompleted) {
      inferenceIdle.complete(Unit)
    }
  }

  val done = CompletableDeferred<Pair<String, String>>()
  val sb = StringBuilder()
  val thinkingSb = StringBuilder()
  LlmChatModelHelper.resetConversation(
    model = model,
    supportImage = model.llmSupportImage,
    supportAudio = model.llmSupportAudio,
    systemInstruction = null,
    tools = emptyList(),
    enableConversationConstrainedDecoding = false,
  )
  LlmChatModelHelper.runInference(
    model = model,
    input = prompt,
    resultListener = { partial, finished, partialThinking ->
      if (partial.startsWith("<ctrl")) {
        if (finished) {
          serviceScope.launch {
            if (!done.isCompleted) {
              done.complete(sb.toString() to thinkingSb.toString())
            }
            markInferenceIdle()
          }
        }
        return@runInference
      }
      if (!partialThinking.isNullOrEmpty()) {
        thinkingSb.append(partialThinking)
      }
      if (partial.isNotEmpty()) {
        llmDebugLog.appendDelta(partial)
        sb.append(partial)
      }
      if (finished) {
        serviceScope.launch {
          if (!done.isCompleted) {
            done.complete(sb.toString() to thinkingSb.toString())
          }
          markInferenceIdle()
        }
      }
    },
    cleanUpListener = {},
    onError = { message ->
      serviceScope.launch {
        llmDebugLog.appendError(message)
        if (!done.isCompleted) {
          done.complete("Error: $message" to thinkingSb.toString())
        }
        markInferenceIdle()
      }
    },
    images = images,
    coroutineScope = serviceScope,
    extraContext = extraContext,
  )
  val (content, reasoning) = done.await()
  return BlockingChatResult(content, reasoning, inferenceIdle)
}

private suspend fun passesSecurity(clientIp: String, securityGateway: SecurityGateway): Boolean {
  var detail = ""
  val allowed =
    when (val decision = securityGateway.evaluate(clientIp)) {
      is Decision.Allow -> {
        detail = "whitelist"
        true
      }
      is Decision.Deny -> {
        detail = "whitelist_disabled"
        false
      }
      is Decision.Pending -> {
        val result = withTimeoutOrNull(30_000) { decision.deferred.await() }
        if (result == null) {
          securityGateway.failPending(clientIp)
          detail = "pending_timeout"
          false
        } else {
          detail = if (result) "pending_allow" else "pending_deny"
          result
        }
      }
    }
  securityGateway.logConnection(clientIp, allowed, detail)
  return allowed
}

private fun verifyBearer(call: ApplicationCall, dataStoreRepository: DataStoreRepository): Boolean {
  val token = dataStoreRepository.readServiceConfig().authToken
  if (token.isNullOrEmpty()) return true
  val header = call.request.headers["Authorization"] ?: return false
  return header == "Bearer $token" || header == token
}
