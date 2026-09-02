plugins { id("com.android.application"); id("org.jetbrains.kotlin.android") }

android {
    namespace = "kr.sht.dangeraudio.onnxbench"
    compileSdk = 35
    defaultConfig { applicationId = "kr.sht.dangeraudio.onnxbench"; minSdk = 26; targetSdk = 35; versionCode = 1; versionName = "0.1" }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures { buildConfig = true }
    aaptOptions { noCompress += setOf("onnx", "f32") }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.29.0")
}
