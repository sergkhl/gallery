package com.google.ai.edge.gallery.inferenceserver

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import androidx.core.content.ContextCompat
import com.google.ai.edge.gallery.di.InferenceBootEntryPoint
import dagger.hilt.android.EntryPointAccessors

class InferenceBootReceiver : BroadcastReceiver() {

  override fun onReceive(context: Context, intent: Intent?) {
    if (intent?.action != Intent.ACTION_BOOT_COMPLETED) return
    val appContext = context.applicationContext
    val entryPoint =
      EntryPointAccessors.fromApplication(appContext, InferenceBootEntryPoint::class.java)
    val repo = entryPoint.dataStoreRepository()
    if (!repo.readServiceConfig().autoStart) return
    ContextCompat.startForegroundService(
      appContext,
      Intent(appContext, InferenceService::class.java),
    )
  }
}
