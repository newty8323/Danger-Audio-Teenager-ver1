package kr.sht.dangeraudio.onnxbench

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.SystemClock
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.ShortBuffer
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Whisper Base ONNX runner optimized for the current Android CPU path.
 *
 * The earlier KV-cache version copied 24 large tensors between Java and ONNX
 * Runtime for every token. On the emulator this was slower than recomputing a
 * short transcript prefix. This runner therefore uses a persistent FP32
 * no-cache decoder and Whisper's original 30-second input window. This keeps
 * the exact accuracy-baseline encoder computation intact.
 */
class WhisperBase(private val context: Context) {
    data class Result(
        val text: String,
        val inputSpeechSeconds: Double,
        val melMs: Double,
        val sessionPrepareMs: Double,
        val encoderMs: Double,
        val decoderMs: Double,
        val tokenCount: Int,
        val stopReason: String,
        val noSpeechProbability: Float,
        val averageLogProbability: Double,
        val warmSession: Boolean,
    )

    private val env = OrtEnvironment.getEnvironment()
    private val sessionOptions = OrtTuning.createOptions()
    private val tokenizer by lazy { WhisperTokenizer(context) }
    private val melFilter by lazy { floatsAsset("whisper_mel_filters.f32", 80 * 201) }
    private val analysisWindow = FloatArray(400) { i -> (0.5 - 0.5 * cos(2.0 * PI * i / 400)).toFloat() }
    private val encoderFile by lazy { assetFile("whisper_encoder_30s.onnx") }
    private val decoderFile by lazy { assetFile("whisper_decoder_30s.onnx") }
    private var encoder: OrtSession? = null
    private var decoder: OrtSession? = null
    private var firstSessionPrepareMs = 0.0

    fun transcribe(stereoVocals: FloatArray): Result {
        var stage = "16 kHz 변환"
        try {
        val speech = prepareSpeechInput(resampleTo16k(stereoVocals))
        if (speech == null) {
            return Result(
                text = "",
                inputSpeechSeconds = 0.0,
                melMs = 0.0,
                sessionPrepareMs = 0.0,
                encoderMs = 0.0,
                decoderMs = 0.0,
                tokenCount = 0,
                stopReason = "silence",
                noSpeechProbability = 1f,
                averageLogProbability = 0.0,
                warmSession = encoder != null && decoder != null,
            )
        }
        stage = "Whisper 모델 자산 복사"
        // Copy large assets before timing DSP so first-run file I/O is not
        // misreported as log-Mel processing time.
        val encoderModel = encoderFile
        val decoderModel = decoderFile
        val melStart = SystemClock.elapsedRealtimeNanos()
        stage = "Whisper log-Mel 생성"
        val features = logMelContext(speech.samples)
        val melMs = (SystemClock.elapsedRealtimeNanos() - melStart) / 1e6

        stage = "Whisper ONNX 세션 준비"
        val hadSessions = encoder != null && decoder != null
        ensureSessions(encoderModel, decoderModel)
        val activeEncoder = requireNotNull(encoder)
        val activeDecoder = requireNotNull(decoder)
                stage = "Whisper encoder ONNX"
                val encoderStart = SystemClock.elapsedRealtimeNanos()
                val hidden = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(features), longArrayOf(1, 80, MEL_FRAMES.toLong())
                ).use { featureTensor ->
                    activeEncoder.run(mapOf("input_features" to featureTensor)).use { output ->
                        val values = FloatArray(ENCODER_FRAMES * 512)
                        (output[0] as OnnxTensor).floatBuffer.get(values)
                        values
                    }
                }
                val encoderMs = (SystemClock.elapsedRealtimeNanos() - encoderStart) / 1e6
                val decoderStart = SystemClock.elapsedRealtimeNanos()
                stage = "Whisper decoder ONNX"
                val decoded = decodeGreedy(activeDecoder, hidden)
                val decoderMs = (SystemClock.elapsedRealtimeNanos() - decoderStart) / 1e6
                stage = "Whisper 토큰 해독"
                val noSpeech = decoded.noSpeechProbability > NO_SPEECH_THRESHOLD &&
                    decoded.averageLogProbability < LOG_PROBABILITY_THRESHOLD
                return Result(
                    if (noSpeech) "" else tokenizer.decode(decoded.ids),
                    speech.activeSeconds,
                    melMs,
                    if (hadSessions) 0.0 else firstSessionPrepareMs,
                    encoderMs,
                    decoderMs,
                    decoded.ids.size - PREFIX.size,
                    if (noSpeech) "no_speech" else decoded.stopReason,
                    decoded.noSpeechProbability,
                    decoded.averageLogProbability,
                    hadSessions,
                )
        } catch (e: Exception) {
            throw IllegalStateException("Whisper 단계 실패: $stage", e)
        }
    }

    private data class DecodeResult(
        val ids: IntArray,
        val stopReason: String,
        val noSpeechProbability: Float,
        val averageLogProbability: Double,
    )

    private data class TokenChoice(val id: Int, val logProbability: Double)

    private fun decodeGreedy(decoder: OrtSession, hidden: FloatArray): DecodeResult {
        val ids = PREFIX.toMutableList()
        val hiddenShape = longArrayOf(1, ENCODER_FRAMES.toLong(), 512)
        var noSpeechProbability = 0f
        var logProbabilitySum = 0.0
        var scoredTokens = 0
        repeat(MAX_NEW_TOKENS) { step ->
            val prefixLength = ids.size
            val choice = OnnxTensor.createTensor(
                env, LongBuffer.wrap(ids.map(Int::toLong).toLongArray()), longArrayOf(1, prefixLength.toLong())
            ).use { inputIds ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(hidden), hiddenShape).use { states ->
                    decoder.run(mapOf("input_ids" to inputIds, "encoder_hidden_states" to states)).use { output ->
                        val logits = (output[0] as OnnxTensor).floatBuffer
                        if (step == 0) {
                            // Whisper defines no-speech confidence at the SOT
                            // position, before the language/task prompt tokens.
                            noSpeechProbability = tokenProbability(logits, SOT_LOGITS_POSITION * VOCAB_SIZE, NO_SPEECH)
                        }
                        argmaxAllowed(logits, (prefixLength - 1) * VOCAB_SIZE)
                    }
                }
            }
            logProbabilitySum += choice.logProbability
            scoredTokens++
            if (choice.id == EOT) {
                return DecodeResult(ids.toIntArray(), "eot", noSpeechProbability, logProbabilitySum / scoredTokens)
            }
            ids += choice.id
            if (hasDegenerateSuffix(ids, PREFIX.size)) {
                return DecodeResult(ids.toIntArray(), "repetition", noSpeechProbability, logProbabilitySum / scoredTokens)
            }
        }
        return DecodeResult(ids.toIntArray(), "token_limit", noSpeechProbability, logProbabilitySum / scoredTokens.coerceAtLeast(1))
    }

    /** Stop only after the generated token stream repeats the same suffix four times. */
    private fun hasDegenerateSuffix(ids: List<Int>, generatedStart: Int): Boolean {
        val generated = ids.size - generatedStart
        for (unitLength in 1..minOf(MAX_REPEAT_UNIT_TOKENS, generated / REPEAT_COUNT)) {
            val repeatedLength = unitLength * REPEAT_COUNT
            val first = ids.size - repeatedLength
            var same = true
            for (offset in unitLength until repeatedLength) {
                if (ids[first + offset] != ids[first + (offset % unitLength)]) {
                    same = false
                    break
                }
            }
            if (same) return true
        }
        return false
    }

    private fun argmaxAllowed(logits: FloatBuffer, offset: Int): TokenChoice {
        var bestId = EOT
        var best = Float.NEGATIVE_INFINITY
        for (id in 0 until VOCAB_SIZE) {
            // No timestamp or control tokens in this transcript-only experiment.
            if (id >= TIMESTAMP_BEGIN || (id in 50257..50363 && id != EOT)) continue
            val value = logits.get(offset + id)
            if (value > best) { best = value; bestId = id }
        }
        var total = 0.0
        for (id in 0 until VOCAB_SIZE) {
            if (id >= TIMESTAMP_BEGIN || (id in 50257..50363 && id != EOT)) continue
            total += kotlin.math.exp((logits.get(offset + id) - best).toDouble())
        }
        return TokenChoice(bestId, -ln(total))
    }

    private fun tokenProbability(logits: FloatBuffer, offset: Int, tokenId: Int): Float {
        var largest = Float.NEGATIVE_INFINITY
        for (id in 0 until VOCAB_SIZE) largest = maxOf(largest, logits.get(offset + id))
        var total = 0.0
        for (id in 0 until VOCAB_SIZE) total += kotlin.math.exp((logits.get(offset + id) - largest).toDouble())
        return (kotlin.math.exp((logits.get(offset + tokenId) - largest).toDouble()) / total).toFloat()
    }

    @Synchronized
    private fun ensureSessions(encoderFile: File, decoderFile: File) {
        if (encoder != null && decoder != null) return
        val start = SystemClock.elapsedRealtimeNanos()
        encoder = env.createSession(encoderFile.absolutePath, sessionOptions)
        decoder = env.createSession(decoderFile.absolutePath, sessionOptions)
        firstSessionPrepareMs = (SystemClock.elapsedRealtimeNanos() - start) / 1e6
    }

    fun close() {
        encoder?.close(); encoder = null
        decoder?.close(); decoder = null
        sessionOptions.close()
    }

    private fun assetFile(name: String): File {
        val target = File(context.filesDir, name)
        val assetLength = context.assets.openFd(name).use { it.length }
        if (!target.exists() || target.length() != assetLength) {
            context.assets.open(name).use { input -> target.outputStream().use(input::copyTo) }
        }
        return target
    }

    /** Resample the mono Demucs stem from 44.1 kHz to Whisper's 16 kHz. */
    private fun resampleTo16k(stereo: FloatArray): FloatArray {
        val sourceSamples = stereo.size / 2
        // Do not multiply these Int values first: 176,400 × 16,000 exceeds
        // Int.MAX_VALUE and previously produced the negative length -33,391.
        val targetSamples = (sourceSamples.toLong() * 16_000L / 44_100L).toInt()
        val result = FloatArray(targetSamples)
        for (i in result.indices) {
            val at = i * 44_100.0 / 16_000.0
            val left = at.toInt().coerceAtMost(sourceSamples - 1)
            val right = (left + 1).coerceAtMost(sourceSamples - 1)
            val mix = (at - left).toFloat()
            result[i] = stereo[left] * (1f - mix) + stereo[right] * mix
        }
        return result
    }

    private data class SpeechInput(val samples: FloatArray, val activeSeconds: Double)

    /**
     * Detect whether the Demucs stem contains speech-like energy without
     * changing its timeline. The previous implementation concatenated only
     * non-silent 20 ms frames. That removed natural pauses and parts of words,
     * creating an unnatural waveform which encouraged Whisper hallucinations.
     */
    private fun prepareSpeechInput(samples: FloatArray): SpeechInput? {
        val frame = 320
        var activeSamples = 0
        for (start in samples.indices step frame) {
            val end = minOf(start + frame, samples.size)
            var energy = 0.0
            for (i in start until end) energy += samples[i] * samples[i]
            val rms = sqrt(energy / (end - start))
            if (rms >= SILENCE_RMS) activeSamples += end - start
        }
        if (activeSamples == 0) return null
        return SpeechInput(samples, activeSamples / 16_000.0)
    }

    private fun logMelContext(speech: FloatArray): FloatArray {
        val input = FloatArray(WINDOW_SECONDS * 16_000)
        speech.copyInto(input, endIndex = minOf(speech.size, input.size))
        val silenceLogMel = ln(1e-10f) / ln(10.0).toFloat()
        // Frames wholly inside zero padding have exactly the clamped silence
        // value. Fill them directly while preserving Whisper's original 30 s
        // tensor shape and encoder input.
        val result = FloatArray(80 * MEL_FRAMES) { silenceLogMel }
        val filter = melFilter
        val powers = FloatArray(201)
        var globalMax = silenceLogMel
        val activeFrames = if (speech.isEmpty()) 0 else {
            ((minOf(speech.size, input.size) + 359) / 160).coerceAtMost(MEL_FRAMES)
        }
        for (frame in 0 until activeFrames) {
            val first = frame * 160 - 200
            WhisperFft400.powerSpectrum(input, first, analysisWindow, powers)
            for (band in 0 until 80) {
                var mel = 0f
                val base = band * 201
                for (bin in 0..200) mel += powers[bin] * melFilter[base + bin]
                val value = (ln(max(mel, 1e-10f)) / ln(10.0).toFloat())
                result[band * MEL_FRAMES + frame] = value
                if (value > globalMax) globalMax = value
            }
        }
        val floor = globalMax - 8f
        for (i in result.indices) result[i] = (max(result[i], floor) + 4f) / 4f
        return result
    }

    private fun floatsAsset(name: String, count: Int): FloatArray {
        val data = context.assets.open(name).readBytes()
        require(data.size == count * 4) { "$name has an unexpected length" }
        return FloatArray(count).also { ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
    }

    companion object {
        const val DECODER_POLICY = "max40_repeat4_no_speech_cached_padding_full_timeline"
        private val PREFIX = intArrayOf(50258, 50264, 50359, 50363)
        private const val EOT = 50257
        private const val NO_SPEECH = 50362
        private const val SOT_LOGITS_POSITION = 0
        private const val TIMESTAMP_BEGIN = 50364
        private const val VOCAB_SIZE = 51865
        // In the one-hour news run every pathological transcript hit the old
        // limit of 128, while 95% of normal four-second windows used <=26.
        private const val MAX_NEW_TOKENS = 40
        private const val REPEAT_COUNT = 4
        private const val MAX_REPEAT_UNIT_TOKENS = 8
        const val NO_SPEECH_THRESHOLD = 0.60f
        const val LOG_PROBABILITY_THRESHOLD = -1.0
        private const val WINDOW_SECONDS = 30
        private const val MEL_FRAMES = WINDOW_SECONDS * 100
        private const val ENCODER_FRAMES = MEL_FRAMES / 2
        private const val SILENCE_RMS = 0.003
        private val CACHE_INPUTS = buildList {
            for (layer in 0 until 6) {
                add("past_self_key_$layer"); add("past_self_value_$layer")
                add("past_cross_key_$layer"); add("past_cross_value_$layer")
            }
        }
    }
}

/**
 * Exact 400-point DFT through a 1,024-point Bluestein FFT.
 *
 * Whisper's mel filters are defined for 400 FFT bins, so zero-padding to 512
 * would change the feature scale. Bluestein preserves the 400-point spectrum
 * while replacing 3000 × 201 × 400 direct DFT operations with radix-2 FFTs.
 */
private object WhisperFft400 {
    private const val N = 400
    private const val M = 1024
    private val chirpRe = FloatArray(N)
    private val chirpIm = FloatArray(N)
    private val kernelRe = FloatArray(M)
    private val kernelIm = FloatArray(M)
    private val workRe = FloatArray(M)
    private val workIm = FloatArray(M)

    init {
        for (i in 0 until N) {
            val angle = Math.PI * i.toDouble() * i / N
            chirpRe[i] = cos(angle).toFloat()
            chirpIm[i] = (-sin(angle)).toFloat()
            kernelRe[i] = cos(angle).toFloat()
            kernelIm[i] = sin(angle).toFloat()
            if (i != 0) {
                kernelRe[M - i] = kernelRe[i]
                kernelIm[M - i] = kernelIm[i]
            }
        }
        fft(kernelRe, kernelIm, inverse = false)
    }

    fun powerSpectrum(samples: FloatArray, first: Int, window: FloatArray, out: FloatArray) {
        java.util.Arrays.fill(workRe, 0f)
        java.util.Arrays.fill(workIm, 0f)
        for (i in 0 until N) {
            val sample = samples[reflect(first + i, samples.size)] * window[i]
            workRe[i] = sample * chirpRe[i]
            workIm[i] = sample * chirpIm[i]
        }
        fft(workRe, workIm, inverse = false)
        for (i in 0 until M) {
            val r = workRe[i] * kernelRe[i] - workIm[i] * kernelIm[i]
            val q = workRe[i] * kernelIm[i] + workIm[i] * kernelRe[i]
            workRe[i] = r; workIm[i] = q
        }
        fft(workRe, workIm, inverse = true)
        for (k in 0..200) {
            val r = workRe[k] * chirpRe[k] - workIm[k] * chirpIm[k]
            val q = workRe[k] * chirpIm[k] + workIm[k] * chirpRe[k]
            out[k] = r * r + q * q
        }
    }

    private fun reflect(value: Int, length: Int): Int {
        var index = value
        while (index < 0 || index >= length) index = if (index < 0) -index else 2 * length - 2 - index
        return index
    }

    private fun fft(re: FloatArray, im: FloatArray, inverse: Boolean) {
        var j = 0
        for (i in 1 until M) {
            var bit = M shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                val tr = re[i]; re[i] = re[j]; re[j] = tr
                val ti = im[i]; im[i] = im[j]; im[j] = ti
            }
        }
        var size = 2
        while (size <= M) {
            val angle = (if (inverse) 2.0 else -2.0) * Math.PI / size
            val stepRe = cos(angle).toFloat(); val stepIm = sin(angle).toFloat()
            for (base in 0 until M step size) {
                var rotRe = 1f; var rotIm = 0f
                for (k in 0 until size / 2) {
                    val even = base + k; val odd = even + size / 2
                    val tr = rotRe * re[odd] - rotIm * im[odd]
                    val ti = rotRe * im[odd] + rotIm * re[odd]
                    re[odd] = re[even] - tr; im[odd] = im[even] - ti
                    re[even] += tr; im[even] += ti
                    val nextRe = rotRe * stepRe - rotIm * stepIm
                    rotIm = rotRe * stepIm + rotIm * stepRe; rotRe = nextRe
                }
            }
            size = size shl 1
        }
        if (inverse) for (i in 0 until M) { re[i] /= M; im[i] /= M }
    }
}

/** Decoder-only side of the Whisper byte-level BPE tokenizer. */
private class WhisperTokenizer(context: Context) {
    private val tokens: Array<String?> = arrayOfNulls(51865)
    private val byteForChar: Map<Char, Byte>

    init {
        val json = JSONObject(context.assets.open("whisper_vocab.json").bufferedReader().readText())
        val iterator = json.keys()
        while (iterator.hasNext()) {
            val piece = iterator.next()
            tokens[json.getInt(piece)] = piece
        }
        byteForChar = buildByteMap()
    }

    fun decode(ids: IntArray): String {
        val bytes = ArrayList<Byte>()
        for (id in ids) {
            if (id >= 50257) continue
            for (ch in tokens[id].orEmpty()) byteForChar[ch]?.let(bytes::add)
        }
        return bytes.toByteArray().toString(Charsets.UTF_8).trim()
    }

    private fun buildByteMap(): Map<Char, Byte> {
        val visible = ArrayList<Int>().apply {
            addAll(33..126); addAll(161..172); addAll(174..255)
        }
        val chars = visible.toMutableList()
        var next = 256
        for (b in 0..255) if (b !in visible) { visible.add(b); chars.add(next++) }
        return visible.indices.associate { chars[it].toChar() to visible[it].toByte() }
    }
}
