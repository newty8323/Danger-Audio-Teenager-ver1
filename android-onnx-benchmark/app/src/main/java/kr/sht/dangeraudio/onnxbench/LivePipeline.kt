package kr.sht.dangeraudio.onnxbench

import android.content.Context
import android.os.SystemClock
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.Base64
import kotlin.concurrent.thread

/** End-to-end local cascade for each new four-second microphone window. */
class LivePipeline(context: Context) : AutoCloseable {
    data class Result(val acoustic: Float, val text: Float, val transcript: String, val alert: Boolean, val elapsedMs: Double)
    private val ced = CedTrigger(context)
    private val demucs = LiveDemucs(context)
    private val whisper = WhisperBase(context)
    private val text = KoElectraHarm(context)

    fun process(window16k: FloatArray, serverUrl: String?): Result {
        val started = SystemClock.elapsedRealtimeNanos()
        val acoustic = ced.score(window16k)
        val vocals = demucs.vocals(window16k)
        val asr = whisper.transcribe(vocals)
        val textScore = text.score(asr.text)
        val alert = acoustic >= ACOUSTIC_THRESHOLD || textScore >= TEXT_THRESHOLD
        val elapsed = (SystemClock.elapsedRealtimeNanos() - started) / 1e6
        val result = Result(acoustic, textScore, asr.text, alert, elapsed)
        if (alert && !serverUrl.isNullOrBlank()) sendToServer(serverUrl, window16k, result)
        return result
    }

    private fun sendToServer(url: String, wave: FloatArray, result: Result) = thread(name = "qwen-escalator", isDaemon = true) {
        runCatching {
            val payload = JSONObject().apply {
                put("clip_id", "android_${System.currentTimeMillis()}")
                put("event", JSONObject().apply {
                    put("start", 0); put("end", 4); put("windows", 1)
                    put("peak_acoustic", result.acoustic); put("peak_text", result.text)
                    put("reasons", if (result.text >= TEXT_THRESHOLD) listOf("text") else listOf("acoustic"))
                    put("transcripts", listOf(result.transcript)); put("peak_score", maxOf(result.acoustic, result.text))
                })
                put("request", JSONObject().apply { put("task", "harm_degree_percent"); put("branches", listOf("acoustic", "text")) })
                put("audio_wav_b64", wavBase64(wave)); put("audio_sr", 16_000)
            }
            val connection = (URL(url).openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"; connectTimeout = 10_000; readTimeout = 30_000
                doOutput = true; setRequestProperty("Content-Type", "application/json")
            }
            connection.outputStream.use { it.write(payload.toString().toByteArray()) }
            connection.inputStream.close(); connection.disconnect()
        }
    }

    private fun wavBase64(wave: FloatArray): String {
        val bytes = ByteArrayOutputStream()
        DataOutputStream(bytes).use { out ->
            fun i(value: Int) { out.write(value); out.write(value ushr 8); out.write(value ushr 16); out.write(value ushr 24) }
            fun s(value: Int) { out.write(value); out.write(value ushr 8) }
            out.writeBytes("RIFF"); i(36 + wave.size * 2); out.writeBytes("WAVEfmt "); i(16); s(1); s(1); i(16_000); i(32_000); s(2); s(16); out.writeBytes("data"); i(wave.size * 2)
            wave.forEach { s((it.coerceIn(-1f, 1f) * 32767).toInt()) }
        }
        return Base64.getEncoder().encodeToString(bytes.toByteArray())
    }
    override fun close() { ced.close(); demucs.close(); whisper.close(); text.close() }
    companion object { const val ACOUSTIC_THRESHOLD = 0.620f; const val TEXT_THRESHOLD = 0.450f }
}
