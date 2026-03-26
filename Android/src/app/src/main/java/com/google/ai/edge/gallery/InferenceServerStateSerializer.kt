package com.google.ai.edge.gallery

import androidx.datastore.core.CorruptionException
import androidx.datastore.core.Serializer
import com.google.ai.edge.gallery.data.Accelerator
import com.google.ai.edge.gallery.proto.InferenceServerState
import com.google.ai.edge.gallery.proto.ServiceConfig
import com.google.protobuf.InvalidProtocolBufferException
import java.io.InputStream
import java.io.OutputStream

object InferenceServerStateSerializer : Serializer<InferenceServerState> {
  private val defaultConfig =
    ServiceConfig.newBuilder()
      .setPort(8080)
      .setAutoStart(false)
      .setAutoStartOnAppOpen(false)
      .setLlmAccelerator(Accelerator.GPU.label)
      .build()

  override val defaultValue: InferenceServerState =
    InferenceServerState.newBuilder().setConfig(defaultConfig).build()

  override suspend fun readFrom(input: InputStream): InferenceServerState {
    try {
      val parsed = InferenceServerState.parseFrom(input)
      if (!parsed.hasConfig()) {
        return parsed.toBuilder().setConfig(defaultConfig).build()
      }
      return parsed
    } catch (exception: InvalidProtocolBufferException) {
      throw CorruptionException("Cannot read inference server proto.", exception)
    }
  }

  override suspend fun writeTo(t: InferenceServerState, output: OutputStream) = t.writeTo(output)
}
