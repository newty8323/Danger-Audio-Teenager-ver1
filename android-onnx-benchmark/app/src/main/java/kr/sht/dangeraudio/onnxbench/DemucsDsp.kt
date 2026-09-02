package kr.sht.dangeraudio.onnxbench

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/** Signal-processing half of the exported Hybrid Demucs graph.
 *
 * The learned layers remain in ONNX. This class mirrors the 4 s fixed-shape
 * PyTorch preprocessing/postprocessing: normalized Hann STFT, mixture-phase
 * reconstruction, normalized Hann ISTFT and the waveform-branch addition.
 */
object DemucsDsp {
    const val sampleRate = 44_100
    const val samples = 176_400
    private const val nfft = 4096
    private const val hop = 1024
    private const val pad = 1536
    private const val frames = 173
    private const val freqs = 2048
    private const val sources = 4
    private const val vocals = 3
    private val window = FloatArray(nfft) { i -> (0.5 - 0.5 * cos(2.0 * PI * i / nfft)).toFloat() }
    private val unit = sqrt(nfft.toDouble()).toFloat()

    data class Spectrum(val magnitude: FloatArray, val real: FloatArray, val imag: FloatArray)
    data class Parts(val frequency: FloatArray, val waveform: FloatArray, val combined: FloatArray)

    fun stft(wave: FloatArray): Spectrum {
        require(wave.size == 2 * samples)
        val magnitude = FloatArray(2 * freqs * frames)
        val real = FloatArray(2 * freqs * frames)
        val imag = FloatArray(2 * freqs * frames)
        val re = FloatArray(nfft)
        val im = FloatArray(nfft)
        val hybridLength = hop * frames + 2 * pad
        for (channel in 0..1) for (frame in 0 until frames) {
            // Stored Demucs frame `frame` corresponds to full STFT frame `frame + 2`.
            val fullFrame = frame + 2
            for (i in 0 until nfft) {
                val centered = fullFrame * hop + i - nfft / 2
                val hybrid = reflect(centered, hybridLength)
                val original = reflect(hybrid - pad, samples)
                re[i] = wave[channel * samples + original] * window[i]
                im[i] = 0f
            }
            fft(re, im, inverse = false)
            for (f in 0 until freqs) {
                val at = (channel * freqs + f) * frames + frame
                val r = re[f] / unit
                val q = im[f] / unit
                real[at] = r; imag[at] = q
                magnitude[at] = sqrt(r * r + q * q)
            }
        }
        return Spectrum(magnitude, real, imag)
    }

    fun reconstruct(spec: Spectrum, sourceMagnitudes: FloatArray, sourceWaveforms: FloatArray): Parts {
        val length = hop * frames + 2 * pad
        // torch.istft(center=true) overlap-adds into a signal with one FFT
        // half-window on each side, then removes the leading half-window.
        // Keeping that 2,048-sample margin here is essential: omitting it
        // shifts the reconstructed vocals by 46 ms.
        val rawLength = length + nfft
        val out = FloatArray(2 * rawLength)
        val norm = FloatArray(rawLength)
        val re = FloatArray(nfft)
        val im = FloatArray(nfft)
        // Production audio is mono duplicated to Demucs stereo.  Use the left
        // stem as the canonical mono reconstruction, then duplicate it. This
        // avoids treating an arbitrary movie's two channels as independent
        // source-separation targets.
        for (channel in 0..1) for (fullFrame in 0 until frames + 4) {
            val modelChannel = 0
            java.util.Arrays.fill(re, 0f); java.util.Arrays.fill(im, 0f)
            val modelFrame = fullFrame - 2
            if (modelFrame in 0 until frames) for (f in 0 until freqs) {
                val mixAt = (modelChannel * freqs + f) * frames + modelFrame
                val sourceAt = (((vocals * 2 + modelChannel) * freqs + f) * frames + modelFrame)
                val scale = sourceMagnitudes[sourceAt] / spec.magnitude[mixAt].coerceAtLeast(1e-8f)
                re[f] = spec.real[mixAt] * scale
                im[f] = spec.imag[mixAt] * scale
                if (f > 0) { re[nfft - f] = re[f]; im[nfft - f] = -im[f] }
            }
            fft(re, im, inverse = true)
            val offset = fullFrame * hop
            for (i in 0 until nfft) {
                val index = offset + i
                if (index < rawLength) {
                    val w = window[i]
                    // `out` contains a full ISTFT margin (rawLength) for
                    // each channel.  Using `length` here made the second
                    // channel begin 4,096 samples too early and overwrite
                    // the tail of the first channel's buffer.
                    out[channel * rawLength + index] += re[i] * unit * w
                    if (channel == 0) norm[index] += w * w
                }
            }
        }
        val frequencyResult = FloatArray(2 * samples)
        val waveformResult = FloatArray(2 * samples)
        val result = FloatArray(2 * samples)
        for (channel in 0..1) for (i in 0 until samples) {
            val outputIndex = nfft / 2 + pad + i
            val frequency = out[channel * rawLength + outputIndex] / norm[outputIndex].coerceAtLeast(1e-8f)
            val waveAt = ((vocals * 2) * samples + i)
            frequencyResult[channel * samples + i] = frequency
            waveformResult[channel * samples + i] = sourceWaveforms[waveAt]
            result[channel * samples + i] = frequency + waveformResult[channel * samples + i]
        }
        return Parts(frequencyResult, waveformResult, result)
    }

    private fun reflect(value: Int, length: Int): Int {
        var index = value
        while (index < 0 || index >= length) index = if (index < 0) -index else 2 * length - 2 - index
        return index
    }

    /** In-place radix-2 FFT. Forward is unnormalised; inverse divides by N. */
    private fun fft(re: FloatArray, im: FloatArray, inverse: Boolean) {
        var j = 0
        for (i in 1 until nfft) {
            var bit = nfft shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) { val tr = re[i]; re[i] = re[j]; re[j] = tr; val ti = im[i]; im[i] = im[j]; im[j] = ti }
        }
        var size = 2
        while (size <= nfft) {
            val angle = (if (inverse) 2.0 else -2.0) * PI / size
            val wrStep = cos(angle).toFloat(); val wiStep = sin(angle).toFloat()
            for (base in 0 until nfft step size) {
                var wr = 1f; var wi = 0f
                for (k in 0 until size / 2) {
                    val even = base + k; val odd = even + size / 2
                    val tr = wr * re[odd] - wi * im[odd]; val ti = wr * im[odd] + wi * re[odd]
                    re[odd] = re[even] - tr; im[odd] = im[even] - ti
                    re[even] += tr; im[even] += ti
                    val nextWr = wr * wrStep - wi * wiStep; wi = wr * wiStep + wi * wrStep; wr = nextWr
                }
            }
            size = size shl 1
        }
        if (inverse) for (i in 0 until nfft) { re[i] /= nfft; im[i] /= nfft }
    }
}
