package com.google.ai.edge.gallery.inferenceserver

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.google.ai.edge.gallery.R
import com.google.ai.edge.gallery.data.DataStoreRepository
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.ui.llmchat.LlmChatModelHelper
import dagger.hilt.android.AndroidEntryPoint
import io.ktor.server.cio.CIO
import io.ktor.server.engine.EmbeddedServer
import io.ktor.server.engine.embeddedServer
import javax.inject.Inject
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

@AndroidEntryPoint
class InferenceService : Service() {

  @Inject lateinit var securityGateway: SecurityGateway
  @Inject lateinit var dataStoreRepository: DataStoreRepository
  @Inject lateinit var inferenceModelRegistry: InferenceModelRegistry
  @Inject lateinit var httpInferenceRuntime: HttpInferenceRuntime

  private val inferenceMutex = Mutex()
  private val job = SupervisorJob()
  private val serviceScope = CoroutineScope(job + Dispatchers.Default)

  /** Bound LLM for this service session; [Model.instance] null when idle-unloaded. */
  private var boundModel: Model? = null
  private var boundPort: Int = 0
  private var server: EmbeddedServer<*, *>? = null
  @Volatile private var startRequested = false

  private val idleScheduleLock = Any()
  private var idleWatchJob: Job? = null
  /** Wall clock of last activity touch; 0 = no unload countdown. */
  @Volatile private var lastInferenceActivityWallMs: Long = 0L

  override fun onCreate() {
    super.onCreate()
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
      val channel =
        NotificationChannel(
          FGS_CHANNEL_ID,
          getString(R.string.inference_service_channel_name),
          NotificationManager.IMPORTANCE_LOW,
        )
      val nm = getSystemService(NotificationManager::class.java)
      nm.createNotificationChannel(channel)
    }
  }

  override fun onBind(intent: Intent?): IBinder? = null

  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    if (intent?.action == ACTION_STOP) {
      stopForeground(STOP_FOREGROUND_REMOVE)
      stopSelf()
      return START_NOT_STICKY
    }

    synchronized(this) {
      if (startRequested) {
        return START_STICKY
      }
      startRequested = true
    }

    val configuredPort =
      dataStoreRepository.readServiceConfig().port.takeIf { it in 1024..65535 }
    startForeground(NOTIFY_ID, buildNotificationStarting(portHint = configuredPort))
    serviceScope.launch {
      try {
        startServerOrThrow()
      } catch (_: Exception) {
        stopSelf()
      }
    }
    return START_STICKY
  }

  private fun cancelIdleWatchLocked() {
    idleWatchJob?.cancel()
    idleWatchJob = null
    lastInferenceActivityWallMs = 0L
  }

  private fun cancelIdleWatch() {
    synchronized(idleScheduleLock) { cancelIdleWatchLocked() }
  }

  /** Resets idle-unload deadline when the model is loaded; no-op if unloaded or feature off. */
  private fun touchInferenceActivity() {
    val model = boundModel ?: return
    if (model.instance == null) return
    val minutes = dataStoreRepository.readServiceConfig().idleUnloadMinutes
    synchronized(idleScheduleLock) {
      cancelIdleWatchLocked()
      if (minutes <= 0) return
      lastInferenceActivityWallMs = System.currentTimeMillis()
      idleWatchJob = serviceScope.launch { idleUnloadWaitLoop() }
    }
  }

  private suspend fun idleUnloadWaitLoop() {
    while (true) {
      val minutes = dataStoreRepository.readServiceConfig().idleUnloadMinutes
      if (minutes <= 0) {
        synchronized(idleScheduleLock) {
          idleWatchJob = null
          lastInferenceActivityWallMs = 0L
        }
        return
      }
      val lastTouch = synchronized(idleScheduleLock) { lastInferenceActivityWallMs }
      if (lastTouch == 0L) return
      val deadline = lastTouch + minutes * 60_000L
      val now = System.currentTimeMillis()
      if (now >= deadline) {
        runUnloadIfStillDue()
        return
      }
      delay(minOf(deadline - now, 60_000L).coerceAtLeast(1L))
    }
  }

  private suspend fun runUnloadIfStillDue() {
    inferenceMutex.withLock {
      val minutes = dataStoreRepository.readServiceConfig().idleUnloadMinutes
      if (minutes <= 0) return@withLock
      val model = boundModel ?: return@withLock
      if (model.instance == null) return@withLock
      val now = System.currentTimeMillis()
      synchronized(idleScheduleLock) {
        val lastTouch = lastInferenceActivityWallMs
        if (lastTouch == 0L) return@withLock
        if (now < lastTouch + minutes * 60_000L) return@withLock
      }
      LlmChatModelHelper.cleanUp(model) {}
      httpInferenceRuntime.setRunning(model.name, boundPort, modelLoaded = false)
      synchronized(idleScheduleLock) {
        lastInferenceActivityWallMs = 0L
        idleWatchJob = null
      }
      Log.d(TAG, "Idle-unloaded model '${model.name}'")
      updateRunningNotification()
    }
  }

  private suspend fun ensureLlmLoadedUnderMutex(): Model {
    val model = boundModel ?: error("HTTP inference service not initialized")
    if (model.instance != null) return model
    val config = dataStoreRepository.readServiceConfig()
    val textAcceleratorLabel =
      HttpInferenceAccelerator.resolveLabel(config.llmAccelerator, model.accelerators)
    suspendCancellableCoroutine { cont ->
      LlmChatModelHelper.initialize(
        context = applicationContext,
        model = model,
        supportImage = model.llmSupportImage,
        supportAudio = model.llmSupportAudio,
        onDone = { error ->
          if (error.isEmpty()) {
            cont.resume(Unit)
          } else {
            cont.resumeWithException(IllegalStateException(error))
          }
        },
        systemInstruction = null,
        tools = emptyList(),
        enableConversationConstrainedDecoding = false,
        coroutineScope = serviceScope,
        textAcceleratorLabel = textAcceleratorLabel,
      )
    }
    httpInferenceRuntime.setRunning(model.name, boundPort, modelLoaded = true)
    updateRunningNotification()
    return model
  }

  private suspend fun startServerOrThrow() {
    val config = dataStoreRepository.readServiceConfig()
    val port = config.port.takeIf { it in 1024..65535 } ?: 8080
    boundPort = port
    val defaultName = config.defaultModelName
    val model =
      (if (defaultName.isNotEmpty()) inferenceModelRegistry.modelByName(defaultName) else null)
        ?: inferenceModelRegistry.downloadedLlmModels.value.firstOrNull()
        ?: error("No downloaded LLM model. Download a model in Models, or set a default in HTTP inference settings.")

    val textAcceleratorLabel =
      HttpInferenceAccelerator.resolveLabel(config.llmAccelerator, model.accelerators)

    boundModel = model

    suspendCancellableCoroutine { cont ->
      LlmChatModelHelper.initialize(
        context = applicationContext,
        model = model,
        supportImage = model.llmSupportImage,
        supportAudio = model.llmSupportAudio,
        onDone = { error ->
          if (error.isEmpty()) {
            cont.resume(Unit)
          } else {
            cont.resumeWithException(IllegalStateException(error))
          }
        },
        systemInstruction = null,
        tools = emptyList(),
        enableConversationConstrainedDecoding = false,
        coroutineScope = serviceScope,
        textAcceleratorLabel = textAcceleratorLabel,
      )
    }

    val embedded =
      embeddedServer(CIO, port = port, host = "0.0.0.0") {
        configureInferenceOpenAiRoutes(
          securityGateway = securityGateway,
          dataStoreRepository = dataStoreRepository,
          inferenceModelRegistry = inferenceModelRegistry,
          inferenceMutex = inferenceMutex,
          serviceScope = serviceScope,
          boundModelName = model.name,
          ensureLlmLoaded = { ensureLlmLoadedUnderMutex() },
          onInferenceActivity = { touchInferenceActivity() },
        )
      }
    embedded.start(wait = false)
    server = embedded
    httpInferenceRuntime.setRunning(model.name, port, modelLoaded = true)
    touchInferenceActivity()
    updateRunningNotification()
  }

  private fun updateRunningNotification() {
    val port = boundPort
    if (port == 0) return
    val modelLoaded = boundModel?.instance != null
    val nm = getSystemService(NotificationManager::class.java)
    nm.notify(NOTIFY_ID, buildNotificationRunning(port, modelLoaded))
  }

  private fun openInferenceSettingsPendingIntent(): PendingIntent {
    val intent =
      Intent(Intent.ACTION_VIEW, Uri.parse(HTTP_INFERENCE_DEEP_LINK)).apply {
        setPackage(packageName)
        addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP)
      }
    return PendingIntent.getActivity(
      this,
      PI_OPEN_SETTINGS,
      intent,
      PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
    )
  }

  private fun stopServerPendingIntent(): PendingIntent {
    val intent =
      Intent(this, InferenceService::class.java).apply { action = ACTION_STOP }
    return PendingIntent.getService(
      this,
      PI_STOP_SERVER,
      intent,
      PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
    )
  }

  private fun buildForegroundNotification(contentText: String): NotificationCompat.Builder {
    return NotificationCompat.Builder(this, FGS_CHANNEL_ID)
      .setSmallIcon(R.mipmap.ic_launcher_monochrome)
      .setContentTitle(getString(R.string.inference_service_notification_title))
      .setContentText(contentText)
      .setStyle(NotificationCompat.BigTextStyle().bigText(contentText))
      .setContentIntent(openInferenceSettingsPendingIntent())
      .addAction(
        0,
        getString(R.string.inference_notification_stop_server),
        stopServerPendingIntent(),
      )
      .setOngoing(true)
      .setOnlyAlertOnce(true)
  }

  private fun buildNotificationStarting(portHint: Int?) =
    buildForegroundNotification(
        contentText =
          buildString {
            append(getString(R.string.inference_service_starting))
            append("\n")
            append(getString(R.string.inference_notification_tap_for_settings))
            if (portHint != null) {
              append("\n")
              append(getString(R.string.inference_service_running_on_port, portHint))
            }
          },
      )
      .build()

  private fun buildNotificationRunning(port: Int, modelLoaded: Boolean) =
    buildForegroundNotification(
        contentText =
          buildString {
            append(getString(R.string.inference_service_running_on_port, port))
            if (!modelLoaded) {
              append("\n")
              append(getString(R.string.inference_service_notification_model_unloaded))
            }
            append("\n")
            append(getString(R.string.inference_notification_tap_for_settings))
          },
      )
      .build()

  override fun onDestroy() {
    cancelIdleWatch()
    job.cancel()
    runCatching {
      server?.stop(gracePeriodMillis = 200L, timeoutMillis = 3000L)
    }
    boundModel?.let { m -> LlmChatModelHelper.cleanUp(m) {} }
    boundModel = null
    server = null
    httpInferenceRuntime.setStopped()
    synchronized(this) { startRequested = false }
    super.onDestroy()
  }

  companion object {
    const val ACTION_STOP = "com.google.ai.edge.gallery.inferenceserver.STOP"

    /** Deep link consumed by [com.google.ai.edge.gallery.ui.navigation.GalleryNavHost]. */
    const val HTTP_INFERENCE_DEEP_LINK = "com.google.ai.edge.gallery://http_inference"

    private const val FGS_CHANNEL_ID = "http_inference_service"
    private const val NOTIFY_ID = 0x48545450 // "HTTP"
    private const val PI_OPEN_SETTINGS = 1001
    private const val PI_STOP_SERVER = 1002
    private const val TAG = "AGInferenceService"
  }
}
