package com.google.ai.edge.gallery.inferenceserver

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class ConnectionDecisionReceiver : BroadcastReceiver() {

  @Inject lateinit var securityGateway: SecurityGateway

  override fun onReceive(context: Context, intent: Intent?) {
    if (intent == null) return
    val ip = intent.getStringExtra(SecurityGateway.EXTRA_CLIENT_IP) ?: return
    when (intent.action) {
      SecurityGateway.ACTION_ALLOW -> securityGateway.resolveDecision(clientIp = ip, allow = true)
      SecurityGateway.ACTION_DENY -> securityGateway.resolveDecision(clientIp = ip, allow = false)
    }
  }
}
