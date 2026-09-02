package kr.sht.dangeraudio.onnxbench

import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import androidx.core.app.ActivityCompat
import java.util.concurrent.CancellationException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

class MainActivity : AppCompatActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var result: TextView; private lateinit var serverUrl: EditText
    private lateinit var choose: Button; private lateinit var start: Button; private lateinit var playback: Button; private lateinit var stop: Button; private lateinit var pipeline: LivePipeline
    private var selected: Uri? = null
    private val analyzing = AtomicBoolean(false)
    private val chooseMedia = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        contentResolver.takePersistableUriPermission(uri, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
        selected = uri; start.isEnabled = true
        result.text = "선택한 미디어\n${uri.lastPathSegment ?: uri}\n\n분석 시작을 누르면 원본 파일을 16 kHz mono·4초 창으로 직접 처리합니다."
    }
    private val requestProjection = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { outcome ->
        if (outcome.resultCode != RESULT_OK || outcome.data == null) { result.text = "재생음 캡처 권한이 허용되지 않았습니다."; return@registerForActivityResult }
        startPlaybackService(outcome.resultCode, outcome.data!!)
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState); setContentView(R.layout.activity_main)
        result = findViewById(R.id.result); serverUrl = findViewById(R.id.serverUrl); choose = findViewById(R.id.chooseButton); start = findViewById(R.id.startButton); playback = findViewById(R.id.playbackButton); stop = findViewById(R.id.stopButton)
        pipeline = LivePipeline(this); choose.setOnClickListener { chooseMedia.launch(arrayOf("audio/*", "video/*")) }; start.setOnClickListener { startAnalysis() }; playback.setOnClickListener { requestPlaybackCapture() }; stop.setOnClickListener { stopAnalysis() }
        result.text = "모델 구성: ${BuildConfig.MODEL_PROFILE}\n\nFP32 기준선과 Demucs INT8 후보 APK는 서로 다른 모델 파일로 만들어집니다."
    }
    private fun startAnalysis() {
        val uri = selected ?: return
        if (!analyzing.compareAndSet(false, true)) return
        choose.isEnabled = false; start.isEnabled = false; stop.isEnabled = true
        val target = serverUrl.text.toString().trim().takeIf { it.isNotBlank() }
        worker.execute {
            val report = runCatching {
                val decoder = MediaFileDecoder(this)
                decoder.decodeWindows(uri, { analyzing.get() }) { index, window ->
                    val r = pipeline.process(window, target)
                    runOnUiThread {
                        result.text = "미디어 ${index}번째 4초 구간 분석 완료\n\nCED-mini 음향 위험도: %.3f\nKoELECTRA 언어 위험도: %.3f\n최종 상태: %s\n처리시간: %.1f ms\nRTF: %.3f\n\n받아쓰기\n%s\n\n%s".format(r.acoustic, r.text, if (r.alert) "ALERT — Qwen 전송 대상" else "SAFE", r.elapsedMs, r.elapsedMs / 4000.0, r.transcript.ifBlank { "(빈 문장)" }, if (target == null) "서버 주소 없음: 기기 내부 판정만 수행" else "서버 주소 설정됨: ALERT일 때 음성·텍스트 전송")
                    }
                }
            }
            runOnUiThread {
                choose.isEnabled = true; start.isEnabled = selected != null; stop.isEnabled = false; analyzing.set(false)
                result.text = report.fold(
                    onSuccess = { summary -> "미디어 분석 완료\n원본: ${summary.sourceRate} Hz · ${summary.sourceChannels}채널\n처리: ${summary.windows}개 4초 구간" },
                    onFailure = { e -> if (e is CancellationException) "미디어 분석을 중지했습니다." else "미디어 분석 실패\n${e.stackTraceToString()}" },
                )
            }
        }
    }
    private fun requestPlaybackCapture() {
        if (android.os.Build.VERSION.SDK_INT < android.os.Build.VERSION_CODES.Q) { result.text = "재생음 캡처는 Android 10 이상에서 지원됩니다."; return }
        if (analyzing.get()) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_AUDIO_PERMISSION); return
        }
        val manager = getSystemService(MediaProjectionManager::class.java)
        requestProjection.launch(manager.createScreenCaptureIntent())
    }
    private fun startPlaybackService(resultCode: Int, data: Intent) {
        if (!analyzing.compareAndSet(false, true)) return
        val target = serverUrl.text.toString().trim().takeIf { it.isNotBlank() }
        val count = AtomicInteger(0)
        PlaybackCaptureBus.onWindow = { window ->
            if (analyzing.get()) {
                val index = count.incrementAndGet()
                worker.execute {
                    if (!analyzing.get()) return@execute
                    val report = runCatching { pipeline.process(window, target) }
                    runOnUiThread { result.text = report.fold(
                        onSuccess = { r -> "재생음 ${index}번째 4초 구간 분석 완료\n\nCED-mini 음향 위험도: %.3f\nKoELECTRA 언어 위험도: %.3f\n최종 상태: %s\n처리시간: %.1f ms\nRTF: %.3f\n\n받아쓰기\n%s\n\n%s".format(r.acoustic, r.text, if (r.alert) "ALERT — Qwen 전송 대상" else "SAFE", r.elapsedMs, r.elapsedMs / 4000.0, r.transcript.ifBlank { "(빈 문장)" }, if (target == null) "서버 주소 없음: 기기 내부 판정만 수행" else "서버 주소 설정됨: ALERT일 때 음성·텍스트 전송") },
                        onFailure = { e -> "재생음 분석 실패\n${e.stackTraceToString()}" }) }
                }
            }
        }
        choose.isEnabled = false; start.isEnabled = false; playback.isEnabled = false; stop.isEnabled = true
        result.text = "기기 재생음을 듣는 중… 4초마다 분석합니다. 알림창의 재생음 캡처 표시가 유지됩니다."
        ContextCompat.startForegroundService(this, PlaybackCaptureService.intent(this, resultCode, data))
    }
    private fun stopAnalysis() {
        analyzing.set(false); stopService(Intent(this, PlaybackCaptureService::class.java)); PlaybackCaptureBus.onWindow = null
        choose.isEnabled = true; start.isEnabled = selected != null; playback.isEnabled = true; stop.isEnabled = false
        result.text = "미디어 분석을 중지했습니다."
    }
    override fun onRequestPermissionsResult(code: Int, permissions: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(code, permissions, results)
        if (code == REQUEST_AUDIO_PERMISSION && results.firstOrNull() == PackageManager.PERMISSION_GRANTED) requestPlaybackCapture()
        else if (code == REQUEST_AUDIO_PERMISSION) result.text = "재생음 캡처에는 오디오 권한이 필요합니다."
    }
    override fun onDestroy() { stopAnalysis(); pipeline.close(); worker.shutdown(); super.onDestroy() }
    companion object { private const val REQUEST_AUDIO_PERMISSION = 101 }
}
