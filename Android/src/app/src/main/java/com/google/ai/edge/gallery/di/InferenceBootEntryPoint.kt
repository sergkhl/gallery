package com.google.ai.edge.gallery.di

import com.google.ai.edge.gallery.data.DataStoreRepository
import dagger.hilt.EntryPoint
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent

@EntryPoint
@InstallIn(SingletonComponent::class)
interface InferenceBootEntryPoint {
  fun dataStoreRepository(): DataStoreRepository
}
