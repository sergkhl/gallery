package com.google.ai.edge.gallery.inferenceserver

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import com.google.ai.edge.gallery.R
import com.google.ai.edge.gallery.data.DataStoreRepository
import com.google.ai.edge.gallery.proto.ConnectionLogEntry
import com.google.ai.edge.gallery.proto.ServiceConfig
import com.google.ai.edge.gallery.proto.WhitelistEntry
import dagger.hilt.android.qualifiers.ApplicationContext
import java.util.concurrent.ConcurrentHashMap
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.CompletableDeferred

@Singleton
class SecurityGateway
@Inject
constructor(
  @ApplicationContext private val context: Context,
  private val dataStoreRepository: DataStoreRepository,
) {

  sealed class Decision {
    data object Allow : Decision()

    data object Deny : Decision()

    data class Pending(val deferred: CompletableDeferred<Boolean>) : Decision()
  }

  private val pendingRequests = ConcurrentHashMap<String, CompletableDeferred<Boolean>>()
  private val notificationManager: NotificationManager =
    context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

  init {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
      val channel =
        NotificationChannel(
          CHANNEL_ID,
          context.getString(R.string.inference_notification_channel_name),
          NotificationManager.IMPORTANCE_HIGH,
        )
      notificationManager.createNotificationChannel(channel)
    }
  }

  fun evaluate(clientIp: String): Decision {
    val whitelist = dataStoreRepository.readServiceConfig().whitelistList
    val entry = whitelist.find { it.ip == clientIp }
    return when {
      entry?.enabled == true -> Decision.Allow
      entry != null && !entry.enabled -> Decision.Deny
      else -> {
        val deferred = CompletableDeferred<Boolean>()
        pendingRequests[clientIp] = deferred
        sendConnectionRequestNotification(clientIp)
        Decision.Pending(deferred)
      }
    }
  }

  fun resolveDecision(clientIp: String, allow: Boolean) {
    pendingRequests.remove(clientIp)?.complete(allow)
    notificationManager.cancel(notificationIdForIp(clientIp))
    if (allow) {
      addOrEnableWhitelistEntry(ip = clientIp, label = "")
    }
  }

  fun cancelPendingNotification(clientIp: String) {
    notificationManager.cancel(notificationIdForIp(clientIp))
  }

  /** Completes a pending decision as denied (e.g. user never responded in time). */
  fun failPending(clientIp: String) {
    pendingRequests.remove(clientIp)?.complete(false)
    cancelPendingNotification(clientIp)
  }

  private fun addOrEnableWhitelistEntry(ip: String, label: String) {
    val c = dataStoreRepository.readServiceConfig()
    val entries = c.whitelistList.toMutableList()
    val idx = entries.indexOfFirst { it.ip == ip }
    val now = System.currentTimeMillis()
    if (idx >= 0) {
      entries[idx] =
        entries[idx].toBuilder().setEnabled(true).setLabel(label.ifEmpty { entries[idx].label }).build()
    } else {
      entries.add(
        WhitelistEntry.newBuilder()
          .setIp(ip)
          .setLabel(label)
          .setEnabled(true)
          .setAddedMs(now)
          .build()
      )
    }
    dataStoreRepository.saveServiceConfig(
      c.toBuilder().clearWhitelist().addAllWhitelist(entries).build()
    )
  }

  private fun sendConnectionRequestNotification(clientIp: String) {
    val allowIntent =
      Intent(context, ConnectionDecisionReceiver::class.java).apply {
        action = ACTION_ALLOW
        putExtra(EXTRA_CLIENT_IP, clientIp)
      }
    val denyIntent =
      Intent(context, ConnectionDecisionReceiver::class.java).apply {
        action = ACTION_DENY
        putExtra(EXTRA_CLIENT_IP, clientIp)
      }
    val flags = PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
    val allowPi =
      PendingIntent.getBroadcast(
        context,
        notificationIdForIp(clientIp) + 1,
        allowIntent,
        flags,
      )
    val denyPi =
      PendingIntent.getBroadcast(
        context,
        notificationIdForIp(clientIp) + 2,
        denyIntent,
        flags,
      )

    val notification: Notification =
      NotificationCompat.Builder(context, CHANNEL_ID)
        .setSmallIcon(R.mipmap.ic_launcher_monochrome)
        .setContentTitle(context.getString(R.string.inference_connection_request_title))
        .setContentText(
          context.getString(R.string.inference_connection_request_body, clientIp)
        )
        .setPriority(NotificationCompat.PRIORITY_HIGH)
        .setCategory(NotificationCompat.CATEGORY_CALL)
        .addAction(0, context.getString(R.string.inference_allow), allowPi)
        .addAction(0, context.getString(R.string.inference_deny), denyPi)
        .setAutoCancel(true)
        .build()

    notificationManager.notify(notificationIdForIp(clientIp), notification)
  }

  fun logConnection(clientIp: String, allowed: Boolean, detail: String) {
    dataStoreRepository.addConnectionLogEntry(
      ConnectionLogEntry.newBuilder()
        .setTimeMs(System.currentTimeMillis())
        .setClientIp(clientIp)
        .setAllowed(allowed)
        .setDetail(detail)
        .build()
    )
  }

  companion object {
    const val CHANNEL_ID = "http_inference_connection"
    const val ACTION_ALLOW = "com.google.ai.edge.gallery.inferenceserver.ALLOW"
    const val ACTION_DENY = "com.google.ai.edge.gallery.inferenceserver.DENY"
    const val EXTRA_CLIENT_IP = "client_ip"

    fun notificationIdForIp(clientIp: String): Int = 0x5000_0000 xor clientIp.hashCode() and 0x0fffFFFF
  }
}
