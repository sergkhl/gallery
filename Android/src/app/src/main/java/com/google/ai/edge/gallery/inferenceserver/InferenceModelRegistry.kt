package com.google.ai.edge.gallery.inferenceserver

import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.data.ModelDownloadStatusType
import com.google.ai.edge.gallery.ui.modelmanager.ModelManagerUiState
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Canonical list of downloaded LLM models for components that cannot use [ModelManagerViewModel]
 * (e.g. [InferenceService]). Kept in sync from the view model UI state.
 */
@Singleton
class InferenceModelRegistry @Inject constructor() {

  private val _downloadedLlmModels = MutableStateFlow<List<Model>>(emptyList())
  val downloadedLlmModels: StateFlow<List<Model>> = _downloadedLlmModels.asStateFlow()

  fun syncFromUiState(state: ModelManagerUiState) {
    val models = LinkedHashMap<String, Model>()
    for (task in state.tasks) {
      for (model in task.models) {
        if (
          model.isLlm &&
            state.modelDownloadStatus[model.name]?.status == ModelDownloadStatusType.SUCCEEDED
        ) {
          models[model.name] = model
        }
      }
    }
    _downloadedLlmModels.value =
      models.values.sortedBy { it.displayName.ifEmpty { it.name } }
  }

  fun modelByName(name: String): Model? = _downloadedLlmModels.value.find { it.name == name }
}
