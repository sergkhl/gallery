# Gallery — notes for assistants

## After Android Kotlin changes

When you have updated Kotlin or Android app code under `Android/`, verify it compiles:

```bash
cd Android/src && ./gradlew :app:compileDebugKotlin
```

Run this from the repository root after substantive edits so compile errors surface before hand-off.
