package kr.sht.dangeraudio.onnxbench

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.LongBuffer
import kotlin.math.exp

/** Local KoELECTRA INT8 harm score with the checkpoint's WordPiece vocabulary. */
class KoElectraHarm(private val context: Context) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val vocab by lazy { loadVocab() }
    private val safeIndex by lazy { loadCats().indexOf("safe") }
    private var session: OrtSession? = null

    fun score(text: String): Float {
        if (text.isBlank()) return 0f
        val ids = encode(text)
        val mask = LongArray(ids.size) { 1L }
        val active = ensureSession()
        return OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, ids.size.toLong())).use { input ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(mask), longArrayOf(1, mask.size.toLong())).use { attention ->
                active.run(mapOf("input_ids" to input, "attention_mask" to attention)).use { output ->
                    val logits = (output[0] as OnnxTensor).floatBuffer
                    val values = FloatArray(logits.remaining()); logits.get(values)
                    val largest = values.maxOrNull() ?: 0f
                    val total = values.sumOf { exp((it - largest).toDouble()) }
                    1f - (exp((values[safeIndex] - largest).toDouble()) / total).toFloat()
                }
            }
        }
    }

    private fun encode(text: String): LongArray {
        val tokens = ArrayList<String>(); tokens += "[CLS]"
        for (word in basicTokens(text)) tokens += wordPieces(word)
        tokens += "[SEP]"
        return tokens.take(MAX_TOKENS).map { (vocab[it] ?: vocab.getValue("[UNK]")).toLong() }.toLongArray()
    }

    private fun basicTokens(text: String): List<String> {
        val normalized = text.replace(Regex("[\\u0000-\\u001F]"), " ").trim()
        return Regex("[가-힣A-Za-z0-9]+|[^\\s가-힣A-Za-z0-9]").findAll(normalized).map { it.value }.toList()
    }

    private fun wordPieces(word: String): List<String> {
        if (word in vocab) return listOf(word)
        val pieces = ArrayList<String>(); var start = 0
        while (start < word.length) {
            var end = word.length; var chosen: String? = null
            while (end > start) {
                val raw = word.substring(start, end)
                val candidate = if (start == 0) raw else "##$raw"
                if (candidate in vocab) { chosen = candidate; break }
                end--
            }
            if (chosen == null) return listOf("[UNK]")
            pieces += chosen; start = end
        }
        return pieces
    }

    @Synchronized private fun ensureSession(): OrtSession {
        session?.let { return it }
        return env.createSession(assetFile("koelectra_harm.int8.onnx").absolutePath, OrtSession.SessionOptions()).also { session = it }
    }

    private fun loadVocab(): Map<String, Int> {
        val root = JSONObject(context.assets.open("koelectra_tokenizer.json").bufferedReader().readText())
        val objectVocab = root.getJSONObject("model").getJSONObject("vocab")
        return objectVocab.keys().asSequence().associateWith { objectVocab.getInt(it) }
    }

    private fun loadCats(): List<String> {
        val array = JSONArray(context.assets.open("koelectra_cats.json").bufferedReader().readText())
        return List(array.length()) { array.getString(it) }
    }

    private fun assetFile(name: String): File {
        val target = File(context.filesDir, name)
        val length = context.assets.openFd(name).use { it.length }
        if (!target.exists() || target.length() != length) context.assets.open(name).use { input -> target.outputStream().use(input::copyTo) }
        return target
    }
    override fun close() { session?.close(); session = null }
    companion object { private const val MAX_TOKENS = 128 }
}
