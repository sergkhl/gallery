package com.google.ai.edge.gallery.inferenceserver.dto

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import com.google.ai.edge.gallery.data.MAX_IMAGE_COUNT
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonPrimitive

/** Max decoded bytes per data-URL image (guard memory). */
private const val MAX_DATA_URL_IMAGE_BYTES = 15 * 1024 * 1024

sealed class HttpInferenceInputParseResult {
  data class Ok(val prompt: String, val images: List<Bitmap>) : HttpInferenceInputParseResult()

  data class Error(val message: String) : HttpInferenceInputParseResult()
}

/**
 * Builds a text prompt and optional images for HTTP inference. Only the **last** message with
 * role `user` may contribute images (OpenAI-style `content` array with `image_url` and `data:`
 * URLs). All other messages are flattened with [ChatMessage.textContent].
 */
fun messagesToHttpInferenceInput(req: ChatCompletionRequest): HttpInferenceInputParseResult {
  val messages = req.messages
  if (messages.isEmpty()) {
    return HttpInferenceInputParseResult.Ok(prompt = "", images = emptyList())
  }

  val lastUserIndex = messages.indexOfLast { it.role.equals("user", ignoreCase = true) }
  val imagesFromLastUser = mutableListOf<Bitmap>()
  val sb = StringBuilder()

  for ((index, m) in messages.withIndex()) {
    if (index > 0) {
      sb.append("\n\n")
    }
    val prefix = "${m.role}: "
    if (index == lastUserIndex && lastUserIndex >= 0) {
      when (val parsed = parseLastUserMessageContent(m.content)) {
        is LastUserContentParse.Ok -> {
          sb.append(prefix).append(parsed.text)
          imagesFromLastUser.addAll(parsed.images)
        }
        is LastUserContentParse.Error ->
          return HttpInferenceInputParseResult.Error(parsed.message)
      }
    } else {
      sb.append(prefix).append(m.textContent())
    }
  }

  return HttpInferenceInputParseResult.Ok(prompt = sb.toString(), images = imagesFromLastUser)
}

private sealed class LastUserContentParse {
  data class Ok(val text: String, val images: List<Bitmap>) : LastUserContentParse()

  data class Error(val message: String) : LastUserContentParse()
}

private fun parseLastUserMessageContent(content: JsonElement?): LastUserContentParse {
  when (content) {
    null -> return LastUserContentParse.Ok("", emptyList())
    is JsonPrimitive ->
      return LastUserContentParse.Ok(text = content.content, images = emptyList())
    is JsonArray -> return parseMultimodalArray(content)
    is JsonObject -> return parseSingleContentPart(content)
    else -> return LastUserContentParse.Ok(text = content.toString(), images = emptyList())
  }
}

private fun parseMultimodalArray(arr: JsonArray): LastUserContentParse {
  val textParts = StringBuilder()
  val dataUrls = mutableListOf<String>()
  for (el in arr) {
    val obj = el as? JsonObject ?: continue
    when (val type = obj["type"]?.jsonPrimitive?.content?.lowercase()) {
      "text" -> {
        val t = obj["text"]?.jsonPrimitive?.content ?: ""
        textParts.append(t)
      }
      "image_url" -> {
        val urlResult = imageUrlStringFromPart(obj) ?: continue
        when (urlResult) {
          is ImageUrlOutcome.Unsupported ->
            return LastUserContentParse.Error(urlResult.reason)
          is ImageUrlOutcome.DataUrl -> dataUrls.add(urlResult.url)
        }
      }
      else -> {
        // Ignore unknown part types (forward-compatible).
      }
    }
  }
  if (dataUrls.size > MAX_IMAGE_COUNT) {
    return LastUserContentParse.Error(
      "At most $MAX_IMAGE_COUNT images are allowed in the last user message."
    )
  }
  val images = mutableListOf<Bitmap>()
  for (url in dataUrls) {
    val bitmap =
      decodeDataUrlBitmap(url)
        ?: run {
          images.forEach { it.recycle() }
          return LastUserContentParse.Error(
            "Invalid or too large data URL image (max ${MAX_DATA_URL_IMAGE_BYTES / (1024 * 1024)} MiB decoded)."
          )
        }
    images.add(bitmap)
  }
  return LastUserContentParse.Ok(text = textParts.toString(), images = images)
}

private fun parseSingleContentPart(obj: JsonObject): LastUserContentParse {
  val type = obj["type"]?.jsonPrimitive?.content?.lowercase()
  return when (type) {
    "text" -> LastUserContentParse.Ok(obj["text"]?.jsonPrimitive?.content ?: "", emptyList())
    "image_url" -> {
      val urlResult = imageUrlStringFromPart(obj) ?: return LastUserContentParse.Ok("", emptyList())
      when (urlResult) {
        is ImageUrlOutcome.Unsupported -> LastUserContentParse.Error(urlResult.reason)
        is ImageUrlOutcome.DataUrl -> {
          val bitmap =
            decodeDataUrlBitmap(urlResult.url)
              ?: return LastUserContentParse.Error(
                "Invalid or too large data URL image (max ${MAX_DATA_URL_IMAGE_BYTES / (1024 * 1024)} MiB decoded)."
              )
          LastUserContentParse.Ok("", listOf(bitmap))
        }
      }
    }
    else -> LastUserContentParse.Ok(obj.toString(), emptyList())
  }
}

/** Recycles bitmaps from a failed HTTP inference request (e.g. model does not support vision). */
fun List<Bitmap>.recycleHttpInferenceImages() {
  for (b in this) {
    if (!b.isRecycled) {
      b.recycle()
    }
  }
}

private sealed class ImageUrlOutcome {
  data class DataUrl(val url: String) : ImageUrlOutcome()

  data class Unsupported(val reason: String) : ImageUrlOutcome()
}

private fun imageUrlStringFromPart(obj: JsonObject): ImageUrlOutcome? {
  val imageUrlEl = obj["image_url"] ?: return null
  val url =
    when (imageUrlEl) {
      is JsonPrimitive -> imageUrlEl.content
      is JsonObject -> imageUrlEl["url"]?.jsonPrimitive?.content ?: return null
      else -> return null
    }
  return classifyImageUrl(url)
}

private fun classifyImageUrl(url: String): ImageUrlOutcome {
  val trimmed = url.trim()
  when {
    trimmed.startsWith("http://", ignoreCase = true) ||
      trimmed.startsWith("https://", ignoreCase = true) ->
      return ImageUrlOutcome.Unsupported(
        "Remote image URLs are not supported. Use data:image/...;base64,... in the last user message."
      )
    trimmed.startsWith("data:", ignoreCase = true) -> return ImageUrlOutcome.DataUrl(trimmed)
    else ->
      return ImageUrlOutcome.Unsupported(
        "Invalid image URL. Only data:image/...;base64,... URLs are supported."
      )
  }
}

private fun decodeDataUrlBitmap(url: String): Bitmap? {
  if (!url.startsWith("data:image/", ignoreCase = true)) {
    return null
  }
  val base64Marker = ";base64,"
  val idx = url.indexOf(base64Marker, ignoreCase = true)
  if (idx < 0) {
    return null
  }
  val b64 = url.substring(idx + base64Marker.length).trim()
  val bytes =
    try {
      Base64.decode(b64, Base64.DEFAULT)
    } catch (_: IllegalArgumentException) {
      return null
    }
  if (bytes.isEmpty() || bytes.size > MAX_DATA_URL_IMAGE_BYTES) {
    return null
  }
  return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}
