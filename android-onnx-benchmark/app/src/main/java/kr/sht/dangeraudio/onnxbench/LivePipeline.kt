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
    data class Result(
        val acoustic: Float,
        val text: Float,
        val transcript: String,
        val alert: Boolean,
        val elapsedMs: Double,
        val cedMs: Double,
        val demucsMs: Double,
        val whisperMs: Double,
        val whisperMelMs: Double,
        val whisperEncoderMs: Double,
        val whisperDecoderMs: Double,
        val koElectraMs: Double,
        val inputSpeechSeconds: Double,
        val tokenCount: Int,
        val whisperStopReason: String,
        val whisperWarmSession: Boolean,
        val serverDispatched: Boolean,
        val requestId: String?,
    )

    data class ServerMetric(
        val requestId: String,
        val elapsedMs: Double,
        val success: Boolean,
        val httpStatus: Int,
        val error: String,
    )

    @Volatile var onServerMetric: ((ServerMetric) -> Unit)? = null
    private val ced = CedTrigger(context)
    private val demucs = LiveDemucs(context)
    private val whisper = WhisperBase(context)
    private val text = KoElectraHarm(context)

    fun process(window16k: FloatArray, serverUrl: String?): Result {
        val started = SystemClock.elapsedRealtimeNanos()
        val cedStarted = SystemClock.elapsedRealtimeNanos()
        val acoustic = ced.score(window16k)
        val cedMs = elapsedSince(cedStarted)
        val demucsStarted = SystemClock.elapsedRealtimeNanos()
        val vocals = demucs.vocals(window16k)
        val demucsMs = elapsedSince(demucsStarted)
        val whisperStarted = SystemClock.elapsedRealtimeNanos()
        val asr = whisper.transcribe(vocals)
        val whisperMs = elapsedSince(whisperStarted)
        val textStarted = SystemClock.elapsedRealtimeNanos()
        val textScore = text.score(asr.text)
        val koElectraMs = elapsedSince(textStarted)
        val alert = acoustic >= ACOUSTIC_THRESHOLD || textScore >= TEXT_THRESHOLD
        val requestId = if (alert && !serverUrl.isNullOrBlank()) {
            "android_${System.currentTimeMillis()}"
        } else null
        val elapsed = elapsedSince(started)
        val result = Result(
            acoustic = acoustic,
            text = textScore,
            transcript = asr.text,
            alert = alert,
            elapsedMs = elapsed,
            cedMs = cedMs,
            demucsMs = demucsMs,
            whisperMs = whisperMs,
            whisperMelMs = asr.melMs,
            whisperEncoderMs = asr.encoderMs,
            whisperDecoderMs = asr.decoderMs,
            koElectraMs = koElectraMs,
            inputSpeechSeconds = asr.inputSpeechSeconds,
            tokenCount = asr.tokenCount,
            whisperStopReason = asr.stopReason,
            whisperWarmSession = asr.warmSession,
            serverDispatched = requestId != null,
            requestId = requestId,
        )
        if (requestId != null) sendToServer(serverUrl!!, window16k, result, requestId)
        return result
    }

    private fun sendToServer(url: String, wave: FloatArray, result: Result, requestId: String) = thread(name = "qwen-escalator", isDaemon = true) {
        val started = SystemClock.elapsedRealtimeNanos()
        var status = -1
        val outcome = runCatching {
            val payload = JSONObject().apply {
                put("clip_id", requestId)
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
            try {
                connection.outputStream.use { it.write(payload.toString().toByteArray()) }
                status = connection.responseCode
                (if (status in 200..299) connection.inputStream else connection.errorStream)?.close()
                check(status in 200..299) { "HTTP $status" }
            } finally {
                connection.disconnect()
            }
        }
        onServerMetric?.invoke(ServerMetric(
            requestId = requestId,
            elapsedMs = elapsedSince(started),
            success = outcome.isSuccess,
            httpStatus = status,
            error = outcome.exceptionOrNull()?.message.orEmpty(),
        ))
    }

    private fun elapsedSince(startedNanos: Long) = (SystemClock.elapsedRealtimeNanos() - startedNanos) / 1e6

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
    override fun close() { onServerMetric = null; ced.close(); demucs.close(); whisper.close(); text.close() }
    companion object { const val ACOUSTIC_THRESHOLD = 0.620f; const val TEXT_THRESHOLD = 0.450f }
}
