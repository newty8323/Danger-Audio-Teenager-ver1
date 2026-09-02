package kr.sht.dangeraudio.onnxbench

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioPlaybackCaptureConfiguration
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import java.util.concurrent.atomic.AtomicBoolean

/** Listener bridge from the foreground capture service to the visible activity. */
object PlaybackCaptureBus { @Volatile var onWindow: ((FloatArray) -> Unit)? = null }

/**
 * Foreground service required by Android 10+ for capturing other apps' media
 * playback. It produces the same 16 kHz mono four-second windows as a file.
 */
class PlaybackCaptureService : Service() {
    private var projection: MediaProjection? = null
    private var recorder: AudioRecord? = null
    private var reader: Thread? = null
    private val running = AtomicBoolean(false)

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val resultCode = intent?.getIntExtra(EXTRA_RESULT_CODE, Int.MIN_VALUE) ?: Int.MIN_VALUE
        val resultData = intent?.parcelableIntent(EXTRA_RESULT_DATA) ?: return START_NOT_STICKY
        startCaptureNotification()
        val manager = getSystemService(MediaProjectionManager::class.java)
        projection = manager.getMediaProjection(resultCode, resultData).also { it.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() { stopSelf() }
        }, null) }
        startAudioRecord(projection ?: return START_NOT_STICKY)
        return START_NOT_STICKY
    }

    private fun startCaptureNotification() {
        val notificationManager = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) notificationManager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Danger Audio 재생음 캡처", NotificationManager.IMPORTANCE_LOW)
        )
        val notification: Notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setContentTitle("Danger Audio가 재생음을 분석 중입니다")
            .setContentText("중지는 앱에서 할 수 있습니다.")
            .setOngoing(true)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION)
        } else startForeground(NOTIFICATION_ID, notification)
    }

    private fun startAudioRecord(mediaProjection: MediaProjection) {
        check(Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) { "재생음 캡처는 Android 10 이상에서 지원됩니다." }
        val config = AudioPlaybackCaptureConfiguration.Builder(mediaProjection)
            .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
            .addMatchingUsage(AudioAttributes.USAGE_GAME)
            .addMatchingUsage(AudioAttributes.USAGE_UNKNOWN)
            .build()
        val format = AudioFormat.Builder().setSampleRate(SR).setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setChannelMask(AudioFormat.CHANNEL_IN_MONO).build()
        val bytes = maxOf(AudioRecord.getMinBufferSize(SR, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT), 8192)
        val record = AudioRecord.Builder().setAudioPlaybackCaptureConfig(config).setAudioFormat(format).setBufferSizeInBytes(bytes).build()
        check(record.state == AudioRecord.STATE_INITIALIZED) { "재생음 캡처기를 초기화하지 못했습니다." }
        recorder = record; running.set(true); record.startRecording()
        reader = Thread({ readLoop(record) }, "danger-audio-playback-capture").also { it.start() }
    }

    private fun readLoop(record: AudioRecord) {
        val scratch = ShortArray(2048); val window = FloatArray(SR * 4); var cursor = 0
        while (running.get()) {
            val count = record.read(scratch, 0, scratch.size, AudioRecord.READ_BLOCKING)
            if (count <= 0) continue
            for (index in 0 until count) {
                window[cursor++] = scratch[index] / 32768f
                if (cursor == window.size) { PlaybackCaptureBus.onWindow?.invoke(window.copyOf()); cursor = 0 }
            }
        }
    }

    override fun onDestroy() {
        running.set(false); recorder?.runCatching { stop() }; recorder?.release(); recorder = null
        reader?.join(1_000); reader = null; projection?.stop(); projection = null
        PlaybackCaptureBus.onWindow = null
        super.onDestroy()
    }
    override fun onBind(intent: Intent?): IBinder? = null

    @Suppress("DEPRECATION") private fun Intent.parcelableIntent(name: String): Intent? =
        if (Build.VERSION.SDK_INT >= 33) getParcelableExtra(name, Intent::class.java) else getParcelableExtra(name)

    companion object {
        const val EXTRA_RESULT_CODE = "result_code"; const val EXTRA_RESULT_DATA = "result_data"
        private const val SR = 16_000; private const val CHANNEL_ID = "playback_capture"; private const val NOTIFICATION_ID = 8770
        fun intent(context: Context, resultCode: Int, data: Intent) = Intent(context, PlaybackCaptureService::class.java).apply {
            putExtra(EXTRA_RESULT_CODE, resultCode); putExtra(EXTRA_RESULT_DATA, data)
        }
    }
}
