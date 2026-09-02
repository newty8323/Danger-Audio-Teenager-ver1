package kr.sht.dangeraudio.onnxbench

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.core.content.ContextCompat
import java.util.concurrent.atomic.AtomicBoolean

class LiveAudioCapture(private val context: Context, private val onWindow: (FloatArray) -> Unit) : AutoCloseable {
    private val running = AtomicBoolean(false)
    private var recorder: AudioRecord? = null
    private var thread: Thread? = null
    @SuppressLint("MissingPermission") fun start() {
        check(ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED)
        if (!running.compareAndSet(false, true)) return
        val size = maxOf(AudioRecord.getMinBufferSize(SR, CHANNEL, FORMAT), 8192)
        val record = AudioRecord(MediaRecorder.AudioSource.MIC, SR, CHANNEL, FORMAT, size)
        check(record.state == AudioRecord.STATE_INITIALIZED)
        recorder = record; record.startRecording()
        thread = Thread({ readLoop(record) }, "danger-audio-capture").also { it.start() }
    }
    private fun readLoop(record: AudioRecord) {
        val scratch = ShortArray(2048); val window = FloatArray(WINDOW_SAMPLES); var cursor = 0
        while (running.get()) {
            val read = record.read(scratch, 0, scratch.size); if (read <= 0) continue
            for (i in 0 until read) { window[cursor++] = scratch[i] / 32768f; if (cursor == window.size) { onWindow(window.copyOf()); cursor = 0 } }
        }
    }
    override fun close() { if (!running.compareAndSet(true, false)) return; recorder?.runCatching { stop() }; recorder?.release(); recorder = null; thread?.join(1000); thread = null }
    companion object { const val SR = 16_000; private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO; private const val FORMAT = AudioFormat.ENCODING_PCM_16BIT; private const val WINDOW_SAMPLES = SR * 4 }
}
