package kr.sht.dangeraudio.onnxbench

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max

/** Android ONNX runner for the trained CED-mini acoustic harm trigger. */
class CedTrigger(private val context: Context) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private val filter by lazy { floatsAsset("ced_mel_filters.f32", 64 * 257) }

    /** Returns the maximum sigmoid probability across the four harm classes. */
    fun score(wave16k: FloatArray): Float {
        val features = CedLogMel.compute(wave16k, filter)
        val active = ensureSession()
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(features), longArrayOf(1, 64, 401)).use { input ->
            active.run(mapOf("log_mel" to input)).use { output ->
                val logits = FloatArray(4)
                (output[0] as OnnxTensor).floatBuffer.get(logits)
                logits.maxOf { 1f / (1f + kotlin.math.exp(-it)) }
            }
        }
    }

    @Synchronized
    private fun ensureSession(): OrtSession {
        session?.let { return it }
        val model = assetFile("ced_mini_vio.onnx")
        return env.createSession(model.absolutePath, OrtSession.SessionOptions()).also { session = it }
    }

    private fun assetFile(name: String): File {
        val target = File(context.filesDir, name)
        val length = context.assets.openFd(name).use { it.length }
        if (!target.exists() || target.length() != length) {
            context.assets.open(name).use { input -> target.outputStream().use(input::copyTo) }
        }
        return target
    }

    private fun floatsAsset(name: String, count: Int): FloatArray {
        val data = context.assets.open(name).readBytes()
        require(data.size == count * 4) { "$name has an unexpected length" }
        return FloatArray(count).also {
            ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it)
        }
    }

    override fun close() { session?.close(); session = null }
}

/** Fixed CED feature extractor: 16 kHz, 64 mel bands, FFT 512, hop 160. */
private object CedLogMel {
    private const val FFT = 512
    private const val HOP = 160
    private const val FRAMES = 401
    private val window = FloatArray(FFT) { i -> (0.5 - 0.5 * cos(2.0 * PI * i / FFT)).toFloat() }

    fun compute(source: FloatArray, filters: FloatArray): FloatArray {
        val input = FloatArray(64_000)
        source.copyInto(input, endIndex = minOf(source.size, input.size))
        val power = FloatArray(257)
        val result = FloatArray(64 * FRAMES)
        var globalMax = Float.NEGATIVE_INFINITY
        for (frame in 0 until FRAMES) {
            CedFft512.power(input, frame * HOP - FFT / 2, window, power)
            for (band in 0 until 64) {
                var value = 0f
                val base = band * 257
                for (bin in 0..256) value += power[bin] * filters[base + bin]
                val db = 10f * (ln(max(value, 1e-10f)) / ln(10.0).toFloat())
                result[band * FRAMES + frame] = db
                if (db > globalMax) globalMax = db
            }
        }
        val floor = globalMax - 120f
        for (i in result.indices) result[i] = max(result[i], floor)
        return result
    }
}

/** In-place radix-2 FFT used only by CED's 512-point fixed front end. */
private object CedFft512 {
    private const val N = 512
    fun power(input: FloatArray, first: Int, window: FloatArray, output: FloatArray) {
        val real = FloatArray(N)
        val imag = FloatArray(N)
        for (i in 0 until N) real[i] = reflected(input, first + i) * window[i]
        var j = 0
        for (i in 1 until N) {
            var bit = N shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                val r = real[i]; real[i] = real[j]; real[j] = r
            }
        }
        var size = 2
        while (size <= N) {
            val angle = -2.0 * PI / size
            val wrStep = cos(angle).toFloat(); val wiStep = kotlin.math.sin(angle).toFloat()
            for (base in 0 until N step size) {
                var wr = 1f; var wi = 0f
                for (offset in 0 until size / 2) {
                    val even = base + offset; val odd = even + size / 2
                    val tr = real[odd] * wr - imag[odd] * wi
                    val ti = real[odd] * wi + imag[odd] * wr
                    real[odd] = real[even] - tr; imag[odd] = imag[even] - ti
                    real[even] += tr; imag[even] += ti
                    val nextWr = wr * wrStep - wi * wiStep
                    wi = wr * wiStep + wi * wrStep; wr = nextWr
                }
            }
            size = size shl 1
        }
        for (i in output.indices) output[i] = real[i] * real[i] + imag[i] * imag[i]
    }

    private fun reflected(values: FloatArray, at: Int): Float {
        var index = at
        while (index < 0 || index >= values.size) {
            index = if (index < 0) -index else 2 * values.size - index - 2
        }
        return values[index]
    }
}
