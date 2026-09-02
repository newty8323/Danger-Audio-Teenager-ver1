package kr.sht.dangeraudio.onnxbench

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.FloatBuffer

class LiveDemucs(private val context: Context) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment(); private var session: OrtSession? = null
    fun vocals(input16k: FloatArray): FloatArray {
        val mono44k = resample(input16k); val stereo = FloatArray(mono44k.size * 2); mono44k.copyInto(stereo); mono44k.copyInto(stereo, mono44k.size)
        val spectrum = DemucsDsp.stft(stereo); val active = ensureSession()
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(spectrum.magnitude), longArrayOf(1, 2, 2048, 173)).use { magnitude ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(stereo), longArrayOf(1, 2, 176400)).use { waveform ->
                active.run(mapOf("mixture_magnitude" to magnitude, "mixture_waveform" to waveform)).use { output ->
                    val magnitudes = FloatArray(4 * 2 * 2048 * 173); val waveforms = FloatArray(4 * 2 * 176400)
                    (output[0] as OnnxTensor).floatBuffer.get(magnitudes); (output[1] as OnnxTensor).floatBuffer.get(waveforms)
                    DemucsDsp.reconstruct(spectrum, magnitudes, waveforms).combined
                }
            }
        }
    }
    @Synchronized private fun ensureSession(): OrtSession { session?.let { return it }; val model = assetFile("demucs_4s.onnx"); return env.createSession(model.absolutePath, OrtSession.SessionOptions()).also { session = it } }
    private fun assetFile(name: String): File { val target = File(context.filesDir, name); val length = context.assets.openFd(name).use { it.length }; if (!target.exists() || target.length() != length) context.assets.open(name).use { input -> target.outputStream().use(input::copyTo) }; return target }
    override fun close() { session?.close(); session = null }
    private fun resample(source: FloatArray): FloatArray = FloatArray(176_400) { i -> val at = i * 16_000.0 / 44_100; val left = at.toInt().coerceIn(0, source.lastIndex); val right = (left + 1).coerceAtMost(source.lastIndex); val frac = at - left; (source[left] * (1 - frac) + source[right] * frac).toFloat() }
}
