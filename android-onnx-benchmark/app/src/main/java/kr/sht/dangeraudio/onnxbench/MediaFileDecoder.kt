package kr.sht.dangeraudio.onnxbench

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CancellationException

/**
 * Decodes an audio file selected through Android's document picker.  Android's
 * media decoder accepts common MP3, M4A/AAC, WAV and video-container audio
 * tracks, then this class makes the same 16 kHz mono four-second windows used
 * by the on-device cascade. No microphone is opened.
 */
class MediaFileDecoder(private val context: Context) {
    data class Summary(val sourceRate: Int, val sourceChannels: Int, val windows: Int)

    fun decodeWindows(
        uri: Uri,
        keepRunning: () -> Boolean,
        onWindow: (index: Int, pcm16k: FloatArray) -> Unit,
    ): Summary {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)
            val track = (0 until extractor.trackCount).firstOrNull { index ->
                extractor.getTrackFormat(index).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
            } ?: error("선택한 미디어에서 오디오 트랙을 찾지 못했습니다.")
            extractor.selectTrack(track)
            val inputFormat = extractor.getTrackFormat(track)
            val mime = inputFormat.getString(MediaFormat.KEY_MIME) ?: error("오디오 형식을 알 수 없습니다.")
            val decoder = MediaCodec.createDecoderByType(mime).also { it.configure(inputFormat, null, null, 0); it.start() }
            codec = decoder

            var sampleRate = inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var channels = inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            var pcmEncoding = AudioFormat.ENCODING_PCM_16BIT
            val assembler = WindowAssembler { index, window -> onWindow(index, window) }
            assembler.configure(sampleRate)
            val info = MediaCodec.BufferInfo()
            var inputEnded = false
            var outputEnded = false

            while (!outputEnded) {
                if (!keepRunning()) throw CancellationException("사용자가 미디어 분석을 중지했습니다.")
                if (!inputEnded) {
                    val inputIndex = decoder.dequeueInputBuffer(TIMEOUT_US)
                    if (inputIndex >= 0) {
                        val buffer = decoder.getInputBuffer(inputIndex) ?: error("오디오 입력 버퍼를 열 수 없습니다.")
                        val size = extractor.readSampleData(buffer, 0)
                        if (size < 0) {
                            decoder.queueInputBuffer(inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputEnded = true
                        } else {
                            decoder.queueInputBuffer(inputIndex, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                when (val outputIndex = decoder.dequeueOutputBuffer(info, TIMEOUT_US)) {
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val output = decoder.outputFormat
                        sampleRate = output.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        channels = output.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        pcmEncoding = if (output.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                            output.getInteger(MediaFormat.KEY_PCM_ENCODING)
                        } else AudioFormat.ENCODING_PCM_16BIT
                        assembler.configure(sampleRate)
                    }
                    MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                    else -> if (outputIndex >= 0) {
                        if (info.size > 0) {
                            val data = decoder.getOutputBuffer(outputIndex) ?: error("오디오 출력 버퍼를 열 수 없습니다.")
                            consumePcm(data, info, channels, pcmEncoding, assembler)
                        }
                        outputEnded = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        decoder.releaseOutputBuffer(outputIndex, false)
                    }
                }
            }
            assembler.finish()
            return Summary(sampleRate, channels, assembler.windowCount)
        } finally {
            codec?.runCatching { stop() }; codec?.release(); extractor.release()
        }
    }

    private fun consumePcm(buffer: ByteBuffer, info: MediaCodec.BufferInfo, channels: Int, encoding: Int, assembler: WindowAssembler) {
        val copy = ByteArray(info.size)
        buffer.position(info.offset); buffer.limit(info.offset + info.size); buffer.get(copy)
        val values = ByteBuffer.wrap(copy).order(ByteOrder.LITTLE_ENDIAN)
        when (encoding) {
            AudioFormat.ENCODING_PCM_FLOAT -> while (values.remaining() >= 4 * channels) {
                var mono = 0f; repeat(channels) { mono += values.float }; assembler.offer(mono / channels)
            }
            else -> while (values.remaining() >= 2 * channels) {
                var mono = 0f; repeat(channels) { mono += values.short / 32768f }; assembler.offer(mono / channels)
            }
        }
    }

    /** Streaming linear resampler, so long media files are never stored fully in RAM. */
    private class WindowAssembler(private val emit: (Int, FloatArray) -> Unit) {
        private var sourceRate = TARGET_SR
        private var sourceIndex = -1L
        private var nextOutputAt = 0.0
        private var previous = 0f
        private var hasPrevious = false
        private var window = FloatArray(WINDOW_SAMPLES)
        private var cursor = 0
        var windowCount = 0; private set

        fun configure(rate: Int) { sourceRate = rate; nextOutputAt = 0.0 }
        fun offer(sample: Float) {
            sourceIndex += 1
            if (!hasPrevious) { previous = sample; hasPrevious = true; append(sample); nextOutputAt = sourceRate.toDouble() / TARGET_SR; return }
            while (nextOutputAt <= sourceIndex.toDouble()) {
                val fraction = (nextOutputAt - (sourceIndex - 1)).toFloat().coerceIn(0f, 1f)
                append(previous + (sample - previous) * fraction)
                nextOutputAt += sourceRate.toDouble() / TARGET_SR
            }
            previous = sample
        }
        fun finish() { if (cursor > 0) { window.copyInto(window, cursor, 0, cursor); emit(++windowCount, window.copyOf()); cursor = 0 } }
        private fun append(sample: Float) { window[cursor++] = sample; if (cursor == window.size) { emit(++windowCount, window); window = FloatArray(WINDOW_SAMPLES); cursor = 0 } }
    }
    companion object { private const val TARGET_SR = 16_000; private const val WINDOW_SAMPLES = TARGET_SR * 4; private const val TIMEOUT_US = 10_000L }
}
