package com.google.ai.edge.gallery.inferenceserver

import com.google.ai.edge.gallery.data.Accelerator

/** Resolves HTTP inference text backend from [ServiceConfig] and [com.google.ai.edge.gallery.data.Model]. */
object HttpInferenceAccelerator {

  /** Stored empty or missing field means GPU (migration / default). */
  fun normalizedSavedLabel(llmAccelerator: String): String =
    llmAccelerator.trim().ifEmpty { Accelerator.GPU.label }

  /**
   * Picks the backend to use: [savedLabel] if allowed, else GPU if allowed, else first of [allowed],
   * else CPU.
   */
  fun resolveLabel(savedLabel: String, allowed: List<Accelerator>): String {
    val normalized = normalizedSavedLabel(savedLabel)
    if (allowed.any { it.label == normalized }) return normalized
    if (allowed.contains(Accelerator.GPU)) return Accelerator.GPU.label
    return allowed.firstOrNull()?.label ?: Accelerator.CPU.label
  }

  /** GPU first, then others in [allowed], stable order for UI. */
  fun displayOrder(allowed: List<Accelerator>): List<Accelerator> {
    if (allowed.isEmpty()) return listOf(Accelerator.CPU, Accelerator.GPU)
    val rest = allowed.filter { it != Accelerator.GPU }
    return listOfNotNull(allowed.find { it == Accelerator.GPU }) + rest
  }
}
