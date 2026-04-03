# TODO

## HTTP inference SSE lifecycle

Bind the streaming pipeline in `writeStreamingChat` (mailbox writer and/or inference cancellation) to the **Ktor `ApplicationCall` / request coroutine** scope so that when the HTTP client disconnects mid-stream, the writer and native inference are cancelled or torn down in line with the connection. Today the single-writer job uses `InferenceService`’s `serviceScope` only.
