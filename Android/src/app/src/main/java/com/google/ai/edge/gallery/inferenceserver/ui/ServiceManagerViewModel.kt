package com.google.ai.edge.gallery.inferenceserver.ui

import android.content.Context
import android.content.Intent
import androidx.core.content.ContextCompat
import com.google.ai.edge.gallery.data.DataStoreRepository
import com.google.ai.edge.gallery.inferenceserver.HttpInferenceAccelerator
import com.google.ai.edge.gallery.inferenceserver.HttpInferenceRuntime
import com.google.ai.edge.gallery.inferenceserver.InferenceModelRegistry
import com.google.ai.edge.gallery.inferenceserver.InferenceService
import com.google.ai.edge.gallery.proto.ServiceConfig
import com.google.ai.edge.gallery.proto.WhitelistEntry
import androidx.lifecycle.ViewModel
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

@HiltViewModel
class ServiceManagerViewModel
@Inject
constructor(
  private val dataStoreRepository: DataStoreRepository,
  private val inferenceModelRegistry: InferenceModelRegistry,
  val httpInferenceRuntime: HttpInferenceRuntime,
  @ApplicationContext private val context: Context,
) : ViewModel() {

  private val _config = MutableStateFlow(dataStoreRepository.readServiceConfig())
  val serviceConfig: StateFlow<ServiceConfig> = _config.asStateFlow()

  val downloadedLlmModels = inferenceModelRegistry.downloadedLlmModels

  private val _connectionLog = MutableStateFlow(dataStoreRepository.readConnectionLog())
  val connectionLog = _connectionLog.asStateFlow()

  fun refreshFromStore() {
    _config.value = dataStoreRepository.readServiceConfig()
    _connectionLog.value = dataStoreRepository.readConnectionLog()
  }

  fun startService() {
    ContextCompat.startForegroundService(
      context,
      Intent(context, InferenceService::class.java),
    )
  }

  fun stopService() {
    ContextCompat.startForegroundService(
      context,
      Intent(context, InferenceService::class.java).apply { action = InferenceService.ACTION_STOP },
    )
  }

  fun savePort(port: Int) {
    val p = port.coerceIn(1024, 65535)
    val c = _config.value.toBuilder().setPort(p).build()
    dataStoreRepository.saveServiceConfig(c)
    _config.value = c
  }

  fun setDefaultModelName(name: String) {
    val b = _config.value.toBuilder().setDefaultModelName(name)
    val model = inferenceModelRegistry.modelByName(name)
    val c =
      if (model != null) {
        val resolved =
          HttpInferenceAccelerator.resolveLabel(_config.value.llmAccelerator, model.accelerators)
        b.setLlmAccelerator(resolved).build()
      } else {
        b.build()
      }
    dataStoreRepository.saveServiceConfig(c)
    _config.value = c
  }

  fun setLlmAccelerator(label: String) {
    val trimmed = label.trim()
    if (trimmed.isEmpty()) return
    val models = downloadedLlmModels.value
    if (models.isEmpty()) return
    val defaultName = _config.value.defaultModelName
    val model =
      (if (defaultName.isNotEmpty()) inferenceModelRegistry.modelByName(defaultName) else null)
        ?: models.firstOrNull()
        ?: return
    if (model.accelerators.none { it.label == trimmed }) return
    val c = _config.value.toBuilder().setLlmAccelerator(trimmed).build()
    dataStoreRepository.saveServiceConfig(c)
    _config.value = c
  }

  fun setAutoStart(enabled: Boolean) {
    val c = _config.value.toBuilder().setAutoStart(enabled).build()
    dataStoreRepository.saveServiceConfig(c)
    _config.value = c
  }

  fun setAutoStartOnAppOpen(enabled: Boolean) {
    val c = _config.value.toBuilder().setAutoStartOnAppOpen(enabled).build()
    dataStoreRepository.saveServiceConfig(c)
    _config.value = c
  }

  fun setAuthToken(token: String) {
    val c = _config.value.toBuilder().setAuthToken(token.trim()).build()
    dataStoreRepository.saveServiceConfig(c)
    _config.value = c
  }

  fun setIdleUnloadMinutes(minutes: Int) {
    val m = minutes.coerceIn(0, 10_080) // 0 = never; max 1 week
    val c = _config.value.toBuilder().setIdleUnloadMinutes(m).build()
    dataStoreRepository.saveServiceConfig(c)
    _config.value = c
  }

  fun addWhitelistEntry(ip: String, label: String) {
    val trimmed = ip.trim()
    if (trimmed.isEmpty()) return
    val c = _config.value
    val entries = c.whitelistList.toMutableList()
    if (entries.any { it.ip == trimmed }) return
    entries.add(
      WhitelistEntry.newBuilder()
        .setIp(trimmed)
        .setLabel(label.trim())
        .setEnabled(true)
        .setAddedMs(System.currentTimeMillis())
        .build()
    )
    val next = c.toBuilder().clearWhitelist().addAllWhitelist(entries).build()
    dataStoreRepository.saveServiceConfig(next)
    _config.value = next
  }

  fun removeWhitelistEntry(ip: String) {
    val c = _config.value
    val entries = c.whitelistList.filter { it.ip != ip }
    val next = c.toBuilder().clearWhitelist().addAllWhitelist(entries).build()
    dataStoreRepository.saveServiceConfig(next)
    _config.value = next
  }

  fun setWhitelistEntryEnabled(ip: String, enabled: Boolean) {
    val c = _config.value
    val entries =
      c.whitelistList.map {
        if (it.ip == ip) it.toBuilder().setEnabled(enabled).build() else it
      }
    val next = c.toBuilder().clearWhitelist().addAllWhitelist(entries).build()
    dataStoreRepository.saveServiceConfig(next)
    _config.value = next
  }
}
