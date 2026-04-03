package com.google.ai.edge.gallery.inferenceserver.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ContentCopy
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.google.ai.edge.gallery.GalleryTopAppBar
import com.google.ai.edge.gallery.R
import com.google.ai.edge.gallery.data.AppBarAction
import com.google.ai.edge.gallery.data.AppBarActionType
import com.google.ai.edge.gallery.inferenceserver.HttpInferenceAccelerator
import com.google.ai.edge.gallery.inferenceserver.HttpInferenceRuntime
import com.google.ai.edge.gallery.inferenceserver.LocalNetworkAddresses
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import kotlinx.coroutines.delay

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ServiceManagerScreen(
  navigateUp: () -> Unit,
  modifier: Modifier = Modifier,
  viewModel: ServiceManagerViewModel = hiltViewModel(),
) {
  val config by viewModel.serviceConfig.collectAsState()
  val runtime by viewModel.httpInferenceRuntime.status.collectAsState()
  val models by viewModel.downloadedLlmModels.collectAsState()
  val log by viewModel.connectionLog.collectAsState()
  val llmDebugText by viewModel.llmDebugLog.collectAsState()

  var portText by remember(config.port) { mutableStateOf(config.port.toString()) }
  var newIp by remember { mutableStateOf("") }
  var newLabel by remember { mutableStateOf("") }
  var authTokenText by remember(config.authToken) { mutableStateOf(config.authToken) }
  var idleUnloadMinutesText by remember { mutableStateOf("") }

  LaunchedEffect(config.idleUnloadMinutes) {
    idleUnloadMinutesText = config.idleUnloadMinutes.toString()
  }

  val isRunning = runtime is HttpInferenceRuntime.Status.Running
  val modelLoaded =
    when (val s = runtime) {
      is HttpInferenceRuntime.Status.Running -> s.modelLoaded
      else -> true
    }
  val activePort =
    when (val s = runtime) {
      is HttpInferenceRuntime.Status.Running -> s.port
      else -> config.port
    }
  var accessUrls by remember { mutableStateOf<List<String>>(emptyList()) }
  val clipboard = LocalClipboardManager.current
  val connectionLogTimeFormatter =
    remember {
      DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss").withZone(ZoneId.systemDefault())
    }

  LaunchedEffect(isRunning, activePort) {
    if (!isRunning) {
      accessUrls = emptyList()
      return@LaunchedEffect
    }
    accessUrls = LocalNetworkAddresses.httpBaseUrls(activePort)
    while (true) {
      delay(5000)
      accessUrls = LocalNetworkAddresses.httpBaseUrls(activePort)
    }
  }

  LaunchedEffect(Unit) {
    while (true) {
      delay(2000)
      viewModel.refreshFromStore()
    }
  }

  Scaffold(
    modifier = modifier,
    topBar = {
      GalleryTopAppBar(
        title = stringResource(R.string.http_inference_title),
        leftAction = AppBarAction(actionType = AppBarActionType.NAVIGATE_UP, actionFn = navigateUp),
      )
    },
  ) { padding ->
    LazyColumn(
      modifier = Modifier.padding(padding).padding(horizontal = 16.dp),
      verticalArrangement = Arrangement.spacedBy(12.dp),
      contentPadding = PaddingValues(vertical = 16.dp),
    ) {
      item {
        val running = runtime is HttpInferenceRuntime.Status.Running
        val port =
          when (val s = runtime) {
            is HttpInferenceRuntime.Status.Running -> s.port
            else -> config.port
          }
        Row(
          modifier = Modifier.fillMaxWidth(),
          horizontalArrangement = Arrangement.SpaceBetween,
          verticalAlignment = Alignment.CenterVertically,
        ) {
          Column(modifier = Modifier.weight(1f)) {
            Text(
              if (running) {
                stringResource(R.string.http_inference_service_running, port)
              } else {
                stringResource(R.string.http_inference_service_stopped)
              },
              style = MaterialTheme.typography.titleMedium,
            )
            if (running && !modelLoaded) {
              Text(
                stringResource(R.string.http_inference_model_unloaded_hint),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 4.dp),
              )
            }
          }
          if (running) {
            Button(onClick = viewModel::stopService) {
              Text(stringResource(R.string.http_inference_stop))
            }
          } else {
            Button(
              onClick = viewModel::startService,
              enabled = models.isNotEmpty(),
            ) {
              Text(stringResource(R.string.http_inference_start))
            }
          }
        }
      }

      if (isRunning) {
        item(key = "access_urls") {
          HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
          Text(
            stringResource(R.string.http_inference_access_urls_label),
            style = MaterialTheme.typography.labelLarge,
          )
          Text(
            stringResource(R.string.http_inference_access_urls_hint),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
          )
          if (accessUrls.isEmpty()) {
            Text(
              stringResource(R.string.http_inference_no_lan_ip, activePort),
              style = MaterialTheme.typography.bodyMedium,
              modifier = Modifier.padding(top = 8.dp),
            )
          } else {
            Column(
              verticalArrangement = Arrangement.spacedBy(4.dp),
              modifier = Modifier.padding(top = 8.dp),
            ) {
              for (url in accessUrls) {
                Row(
                  modifier = Modifier.fillMaxWidth(),
                  verticalAlignment = Alignment.CenterVertically,
                ) {
                  SelectionContainer(modifier = Modifier.weight(1f)) {
                    Text(
                      text = url,
                      style = MaterialTheme.typography.bodyLarge,
                      color = MaterialTheme.colorScheme.primary,
                    )
                  }
                  IconButton(
                    onClick = { clipboard.setText(AnnotatedString(url)) },
                  ) {
                    Icon(
                      Icons.Rounded.ContentCopy,
                      contentDescription = stringResource(R.string.http_inference_copy_url),
                    )
                  }
                }
              }
            }
          }
        }

        item(key = "llm_debug_log") {
          val debugScrollState = rememberScrollState()
          LaunchedEffect(debugScrollState) {
            snapshotFlow { llmDebugText }
              .collect {
                delay(16)
                debugScrollState.scrollTo(debugScrollState.maxValue)
              }
          }
          HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
          Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
          ) {
            Text(
              stringResource(R.string.http_inference_llm_debug_log),
              style = MaterialTheme.typography.labelLarge,
            )
            TextButton(onClick = viewModel::clearLlmDebugLog) {
              Text(stringResource(R.string.http_inference_llm_debug_clear))
            }
          }
          Text(
            stringResource(R.string.http_inference_llm_debug_hint),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
          )
          SelectionContainer(
            modifier =
              Modifier
                .fillMaxWidth()
                .padding(top = 8.dp)
                .heightIn(max = 240.dp)
                .verticalScroll(debugScrollState),
          ) {
            Text(
              text =
                if (llmDebugText.isEmpty()) {
                  stringResource(R.string.http_inference_llm_debug_empty)
                } else {
                  llmDebugText
                },
              style = MaterialTheme.typography.bodySmall,
              fontFamily = FontFamily.Monospace,
            )
          }
        }
      }

      item {
        Text(stringResource(R.string.http_inference_port), style = MaterialTheme.typography.labelLarge)
        Row(
          verticalAlignment = Alignment.CenterVertically,
          horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
          OutlinedTextField(
            value = portText,
            onValueChange = { portText = it.filter { ch -> ch.isDigit() }.take(5) },
            modifier = Modifier.weight(1f),
            singleLine = true,
          )
          TextButton(
            onClick = {
              portText.toIntOrNull()?.let { viewModel.savePort(it) }
            }
          ) {
            Text(stringResource(R.string.http_inference_save))
          }
        }
      }

      item {
        Text(
          stringResource(R.string.http_inference_default_model),
          style = MaterialTheme.typography.labelLarge,
        )
        val names = models.map { it.name to it.displayName.ifEmpty { it.name } }
        if (names.isEmpty()) {
          Text(
            stringResource(R.string.http_inference_download_llm_first),
            style = MaterialTheme.typography.bodyMedium,
          )
        } else {
          val effectiveDefault =
            config.defaultModelName.ifEmpty { names.first().first }
          Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            for ((name, label) in names) {
              val selected = effectiveDefault == name
              TextButton(
                onClick = { viewModel.setDefaultModelName(name) },
                modifier = Modifier.fillMaxWidth(),
              ) {
                Text(
                  text = if (selected) "✓ $label" else label,
                  modifier = Modifier.fillMaxWidth(),
                )
              }
            }
          }
        }
      }

      item {
        val namePairs = models.map { it.name to it.displayName.ifEmpty { it.name } }
        Text(
          stringResource(R.string.http_inference_llm_accelerator),
          style = MaterialTheme.typography.labelLarge,
        )
        if (namePairs.isEmpty()) {
          Text(
            stringResource(R.string.http_inference_download_llm_first),
            style = MaterialTheme.typography.bodyMedium,
          )
        } else {
          val effectiveDefaultName =
            config.defaultModelName.ifEmpty { namePairs.first().first }
          val defaultModel =
            models.find { it.name == effectiveDefaultName } ?: models.firstOrNull()
          val allowed = defaultModel?.accelerators.orEmpty()
          val ordered = HttpInferenceAccelerator.displayOrder(allowed)
          val effectiveLabel =
            HttpInferenceAccelerator.resolveLabel(config.llmAccelerator, allowed)
          if (isRunning) {
            Text(
              stringResource(R.string.http_inference_accelerator_stop_to_change),
              style = MaterialTheme.typography.bodySmall,
              color = MaterialTheme.colorScheme.onSurfaceVariant,
              modifier = Modifier.padding(bottom = 4.dp),
            )
          }
          Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            for (acc in ordered) {
              val selected = effectiveLabel == acc.label
              TextButton(
                onClick = { viewModel.setLlmAccelerator(acc.label) },
                enabled = !isRunning,
                modifier = Modifier.fillMaxWidth(),
              ) {
                Text(
                  text = if (selected) "✓ ${acc.label}" else acc.label,
                  modifier = Modifier.fillMaxWidth(),
                )
              }
            }
          }
        }
      }

      item {
        Row(
          modifier = Modifier.fillMaxWidth(),
          horizontalArrangement = Arrangement.SpaceBetween,
          verticalAlignment = Alignment.CenterVertically,
        ) {
          Text(stringResource(R.string.http_inference_auto_start))
          Switch(checked = config.autoStart, onCheckedChange = viewModel::setAutoStart)
        }
      }

      item {
        Column(modifier = Modifier.fillMaxWidth()) {
          Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
          ) {
            Text(stringResource(R.string.http_inference_auto_start_on_app_open))
            Switch(
              checked = config.autoStartOnAppOpen,
              onCheckedChange = viewModel::setAutoStartOnAppOpen,
            )
          }
          Text(
            stringResource(R.string.http_inference_auto_start_on_app_open_hint),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 4.dp),
          )
        }
      }

      item {
        Text(
          stringResource(R.string.http_inference_idle_unload_minutes),
          style = MaterialTheme.typography.labelLarge,
        )
        Row(
          verticalAlignment = Alignment.CenterVertically,
          horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
          OutlinedTextField(
            value = idleUnloadMinutesText,
            onValueChange = { idleUnloadMinutesText = it.filter { ch -> ch.isDigit() }.take(5) },
            modifier = Modifier.weight(1f),
            singleLine = true,
          )
          TextButton(
            onClick = {
              idleUnloadMinutesText.toIntOrNull()?.let { viewModel.setIdleUnloadMinutes(it) }
            }
          ) {
            Text(stringResource(R.string.http_inference_save))
          }
        }
        Text(
          stringResource(R.string.http_inference_idle_unload_hint),
          style = MaterialTheme.typography.bodySmall,
          color = MaterialTheme.colorScheme.onSurfaceVariant,
          modifier = Modifier.padding(top = 4.dp),
        )
      }

      item {
        Column(modifier = Modifier.fillMaxWidth()) {
          Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
          ) {
            Text(stringResource(R.string.http_inference_enable_thinking))
            Switch(
              checked = config.enableThinking,
              onCheckedChange = viewModel::setEnableThinking,
            )
          }
          Text(
            stringResource(R.string.http_inference_enable_thinking_hint),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 4.dp),
          )
        }
      }

      item {
        Text(stringResource(R.string.http_inference_auth_token), style = MaterialTheme.typography.labelLarge)
        OutlinedTextField(
          value = authTokenText,
          onValueChange = { authTokenText = it },
          modifier = Modifier.fillMaxWidth(),
          singleLine = true,
          placeholder = { Text(stringResource(R.string.http_inference_token_hint)) },
        )
        TextButton(onClick = { viewModel.setAuthToken(authTokenText) }) {
          Text(stringResource(R.string.http_inference_save))
        }
      }

      item {
        HorizontalDivider()
        Text(stringResource(R.string.http_inference_whitelist), style = MaterialTheme.typography.titleSmall)
        OutlinedTextField(
          value = newIp,
          onValueChange = { newIp = it },
          modifier = Modifier.fillMaxWidth(),
          placeholder = { Text(stringResource(R.string.http_inference_ip_hint)) },
          singleLine = true,
        )
        OutlinedTextField(
          value = newLabel,
          onValueChange = { newLabel = it },
          modifier = Modifier.fillMaxWidth(),
          placeholder = { Text("Label (optional)") },
          singleLine = true,
        )
        TextButton(
          onClick = {
            viewModel.addWhitelistEntry(newIp, newLabel)
            newIp = ""
            newLabel = ""
          }
        ) {
          Text(stringResource(R.string.http_inference_add_client))
        }
      }

      items(config.whitelistList, key = { it.ip }) { entry ->
        Row(
          modifier = Modifier.fillMaxWidth(),
          horizontalArrangement = Arrangement.SpaceBetween,
          verticalAlignment = Alignment.CenterVertically,
        ) {
          Column(modifier = Modifier.weight(1f)) {
            Text(entry.ip, style = MaterialTheme.typography.bodyLarge)
            if (entry.label.isNotEmpty()) {
              Text(entry.label, style = MaterialTheme.typography.bodySmall)
            }
          }
          Switch(
            checked = entry.enabled,
            onCheckedChange = { viewModel.setWhitelistEntryEnabled(entry.ip, it) },
          )
          TextButton(onClick = { viewModel.removeWhitelistEntry(entry.ip) }) {
            Text(stringResource(R.string.delete))
          }
        }
      }

      item {
        HorizontalDivider()
        Text(stringResource(R.string.http_inference_connection_log), style = MaterialTheme.typography.titleSmall)
      }

      items(log, key = { "${it.timeMs}_${it.clientIp}_${it.detail}" }) { entry ->
        val timeLabel =
          if (entry.timeMs > 0) {
            connectionLogTimeFormatter.format(Instant.ofEpochMilli(entry.timeMs))
          } else {
            "—"
          }
        Text(
          "$timeLabel · ${entry.clientIp} — ${entry.detail} (${if (entry.allowed) "allow" else "deny"})",
          style = MaterialTheme.typography.bodySmall,
        )
      }
    }
  }
}
