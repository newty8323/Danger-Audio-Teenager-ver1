plugins { id("com.android.application"); id("org.jetbrains.kotlin.android") }

// `./gradlew assembleDebug -PmodelProfile=baseline` keeps the verified FP32
// Demucs reference. `-PmodelProfile=demucs-fp16` makes a separate APK whose
// only Demucs asset is the verified FP16 candidate, renamed to the same runtime
// filename. The two models are never bundled together in a release APK.
val modelProfile = providers.gradleProperty("modelProfile").orElse("baseline").get()
require(modelProfile in setOf("baseline", "demucs-fp16", "demucs-int8")) {
    "modelProfile must be baseline, demucs-fp16, or demucs-int8 (was $modelProfile)"
}
val runtimeAssets = layout.buildDirectory.dir("generated/runtime-assets/$modelProfile")
val liveAssets = listOf(
    "cascade_thresholds.json",
    "ced_mel_filters.f32",
    "ced_mini_vio.onnx",
    "koelectra_cats.json",
    "koelectra_harm.int8.onnx",
    "koelectra_tokenizer.json",
    "whisper_decoder_30s.onnx",
    "whisper_encoder_30s.onnx",
    "whisper_mel_filters.f32",
    "whisper_vocab.json",
)
val stageRuntimeAssets by tasks.registering(Sync::class) {
    from("src/main/assets") { include(liveAssets + "demucs_4s.onnx") }
    into(runtimeAssets)
    if (modelProfile != "baseline") {
        exclude("demucs_4s.onnx")
        val source = if (modelProfile == "demucs-fp16") {
            "src/main/assets/hybrid_core_4s.fp16.onnx"
        } else {
            "src/main/assets/demucs_4s.int8.onnx"
        }
        from(source) { rename { "demucs_4s.onnx" } }
        doFirst {
            check(file(source).isFile) {
                "$modelProfile Demucs candidate is missing: $source"
            }
        }
    }
}

android {
    namespace = "kr.sht.dangeraudio.onnxbench"
    compileSdk = 35
    defaultConfig {
        applicationId = "kr.sht.dangeraudio.onnxbench"; minSdk = 26; targetSdk = 35; versionCode = 1; versionName = "0.1"
        buildConfigField("String", "MODEL_PROFILE", "\"$modelProfile\"")
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures { buildConfig = true }
    aaptOptions { noCompress += setOf("onnx", "f32") }
    // Replace (rather than add to) src/main/assets.  The staging task is the
    // sole asset source, so old benchmark fixtures and the other Demucs model
    // cannot accidentally be packaged beside the selected runtime model.
    sourceSets.getByName("main").assets.setSrcDirs(listOf(runtimeAssets))
}

tasks.matching { it.name.startsWith("merge") && it.name.endsWith("Assets") }.configureEach {
    dependsOn(stageRuntimeAssets)
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.29.0")
}
