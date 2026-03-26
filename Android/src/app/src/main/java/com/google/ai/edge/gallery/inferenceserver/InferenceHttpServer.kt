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
            ) { chunk ->
              write(chunk)
              flush()
            }
          }
        } else {
          val (text, idle) =
            runBlockingChat(
              serviceScope = serviceScope,
              model = boundModel,
              prompt = prompt,
              images = images,
              llmDebugLog = llmDebugLog,
            )
          inferenceIdle = idle
          call.respond(
            ChatCompletionResponse(
              id = "chatcmpl-${UUID.randomUUID()}",
              model = boundModel.name,
              choices =
                listOf(
                  ChatCompletionChoice(
                    message = ChatCompletionMessage(content = text)
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

private suspend fun writeStreamingChat(
  serviceScope: CoroutineScope,
  model: Model,
  prompt: String,
  images: List<Bitmap>,
  streamId: String,
  modelName: String,
  inferenceIdle: CompletableDeferred<Unit>,
  llmDebugLog: HttpInferenceLlmDebugLog,
  emit: suspend (String) -> Unit,
) {
  fun markInferenceIdle() {
    if (!inferenceIdle.isCompleted) {
      inferenceIdle.complete(Unit)
    }
  }

  val out = Channel<String>(Channel.UNLIMITED)
  val mailbox = Channel<HttpInferenceStreamEvent>(Channel.UNLIMITED)
  serviceScope.launch {
    try {
      for (event in mailbox) {
        when (event) {
          is HttpInferenceStreamEvent.Token -> {
            if (event.text.isNotEmpty()) {
              llmDebugLog.appendDelta(event.text)
              out.send(event.text)
            }
          }
          is HttpInferenceStreamEvent.Error -> {
            llmDebugLog.appendError(event.message)
            out.send("__ERROR__:${event.message}")
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
    resultListener = { partial, done, _ ->
      if (partial.startsWith("<ctrl")) {
        if (done) {
          mailbox.trySend(HttpInferenceStreamEvent.End)
        }
        return@runInference
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
  )
  for (delta in out) {
    if (delta.startsWith("__ERROR__:")) {
      val err = delta.removePrefix("__ERROR__:")
      emit(
        "data: {\"error\":{\"message\":\"${err.replace("\"", "\\\"")}\"}}\n\n"
      )
      emit("data: [DONE]\n\n")
      return
    }
    val chunk =
      ChatCompletionChunk(
        id = streamId,
        model = modelName,
        choices =
          listOf(
            ChatCompletionChunkChoice(
              delta = ChatCompletionDelta(content = delta),
              finishReason = null,
            )
          ),
      )
    emit("data: ${openAiJson.encodeToString(ChatCompletionChunk.serializer(), chunk)}\n\n")
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

private suspend fun runBlockingChat(
  serviceScope: CoroutineScope,
  model: Model,
  prompt: String,
  images: List<Bitmap>,
  llmDebugLog: HttpInferenceLlmDebugLog,
): Pair<String, CompletableDeferred<Unit>> {
  val inferenceIdle = CompletableDeferred<Unit>()
  fun markInferenceIdle() {
    if (!inferenceIdle.isCompleted) {
      inferenceIdle.complete(Unit)
    }
  }

  val done = CompletableDeferred<String>()
  val sb = StringBuilder()
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
    resultListener = { partial, finished, _ ->
      if (partial.startsWith("<ctrl")) {
        if (finished) {
          serviceScope.launch {
            if (!done.isCompleted) {
              done.complete(sb.toString())
            }
            markInferenceIdle()
          }
        }
        return@runInference
      }
      if (partial.isNotEmpty()) {
        llmDebugLog.appendDelta(partial)
        sb.append(partial)
      }
      if (finished) {
        serviceScope.launch {
          if (!done.isCompleted) {
            done.complete(sb.toString())
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
          done.complete("Error: $message")
        }
        markInferenceIdle()
      }
    },
    images = images,
    coroutineScope = serviceScope,
  )
  return done.await() to inferenceIdle
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
