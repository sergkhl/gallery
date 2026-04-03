package com.google.ai.edge.gallery.inferenceserver.dto

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive

@Serializable
data class ChatCompletionRequest(
  val model: String,
  val messages: List<ChatMessage> = emptyList(),
  val stream: Boolean = false,
  @SerialName("max_tokens") val maxTokens: Int? = null,
  val temperature: Double? = null,
  @SerialName("enable_thinking") val enableThinking: Boolean? = null,
)

@Serializable
data class ChatMessage(val role: String, val content: JsonElement? = null) {
  fun textContent(): String =
    when (content) {
      is JsonPrimitive -> content.content
      null -> ""
      is JsonArray ->
        content
          .filterIsInstance<JsonPrimitive>()
          .joinToString("") { it.content }
      else -> content.toString()
    }
}

@Serializable
data class ModelsListResponse(
  @SerialName("object") val objectType: String = "list",
  val data: List<OpenAiModelInfo>,
)

@Serializable
data class OpenAiModelInfo(
  val id: String,
  @SerialName("object") val objectType: String = "model",
  @SerialName("owned_by") val ownedBy: String = "local",
)

@Serializable
data class ChatCompletionResponse(
  val id: String,
  @SerialName("object") val objectType: String = "chat.completion",
  val model: String,
  val choices: List<ChatCompletionChoice>,
)

@Serializable
data class ChatCompletionChoice(
  val index: Int = 0,
  val message: ChatCompletionMessage,
  @SerialName("finish_reason") val finishReason: String = "stop",
)

@Serializable
data class ChatCompletionMessage(
  val role: String = "assistant",
  val content: String,
  @SerialName("reasoning_content") val reasoningContent: String? = null,
)

@Serializable
data class ChatCompletionChunk(
  val id: String,
  @SerialName("object") val objectType: String = "chat.completion.chunk",
  val model: String,
  val choices: List<ChatCompletionChunkChoice>,
)

@Serializable
data class ChatCompletionChunkChoice(
  val index: Int = 0,
  val delta: ChatCompletionDelta,
  @SerialName("finish_reason") val finishReason: String? = null,
)

@Serializable
data class ChatCompletionDelta(
  val content: String? = null,
  @SerialName("reasoning_content") val reasoningContent: String? = null,
)
