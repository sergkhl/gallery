# Development notes

## Build app locally

To successfully build and run the application through Android Studio, you need to configure it with your own HuggingFace Developer Application ([official doc](https://huggingface.co/docs/hub/oauth#creating-an-oauth-app)). This is required for the model download functionality to work correctly.

Gradle **fails** if OAuth values are missing: there are no fallbacks in source.

After you've created a developer application:

1. In the Gradle project root [`Android/src/`](Android/src/), ensure `local.properties` exists. If Android Studio has not created it yet, copy [`Android/src/local.properties.example`](Android/src/local.properties.example) to `local.properties` and set `sdk.dir` to your Android SDK path if needed.

2. In that same `local.properties`, set:
   - `hf.oauth.clientId` — from your Hugging Face OAuth app
   - `hf.oauth.redirectUri` — the full redirect URI registered there (e.g. `com.example.myapp:/oauth`). The custom URL scheme before `:` must match your Hugging Face redirect URL; it is applied to the manifest automatically.

Do not commit `local.properties`; it is listed in [`.gitignore`](.gitignore).
