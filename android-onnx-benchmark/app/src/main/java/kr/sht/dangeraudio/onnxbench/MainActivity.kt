package kr.sht.dangeraudio.onnxbench

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.util.concurrent.CancellationException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

class MainActivity : AppCompatActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var result: TextView
    private lateinit var serverUrl: EditText
    private lateinit var durationMinutes: EditText
    private lateinit var contentName: EditText
    private lateinit var contentType: Spinner
    private lateinit var choose: Button
    private lateinit var start: Button
    private lateinit var playback: Button
    private lateinit var benchmarkButton: Button
    private lateinit var stop: Button
    private lateinit var shareButton: Button
    private lateinit var pipeline: LivePipeline
    private var selected: Uri? = null
    private val analyzing = AtomicBoolean(false)
    private val submittedWindows = AtomicInteger(0)
    private val completedWindows = AtomicInteger(0)
    private var pendingBenchmark: BenchmarkConfig? = null
    private var benchmark: ContinuousBenchmark? = null
    private var lastBenchmark: BenchmarkFiles? = null

    private val chooseMedia = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        selected = uri
        start.isEnabled = true
        result.text = "선택한 미디어\n${uri.lastPathSegment ?: uri}\n\n분석 시작을 누르면 원본 파일을 16 kHz mono·4초 창으로 직접 처리합니다."
    }

    private val requestProjection = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { outcome ->
        if (outcome.resultCode != RESULT_OK || outcome.data == null) {
            pendingBenchmark = null
            result.text = "재생음 캡처 권한이 허용되지 않았습니다."
            return@registerForActivityResult
        }
        val config = pendingBenchmark
        pendingBenchmark = null
        startPlaybackService(outcome.resultCode, outcome.data!!, config)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        result = findViewById(R.id.result)
        serverUrl = findViewById(R.id.serverUrl)
        durationMinutes = findViewById(R.id.durationMinutes)
        contentName = findViewById(R.id.contentName)
        contentType = findViewById(R.id.contentTypeSpinner)
        choose = findViewById(R.id.chooseButton)
        start = findViewById(R.id.startButton)
        playback = findViewById(R.id.playbackButton)
        benchmarkButton = findViewById(R.id.benchmarkButton)
        stop = findViewById(R.id.stopButton)
        shareButton = findViewById(R.id.shareButton)
        contentType.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            listOf("뉴스", "영화·드라마", "음악·랩", "게임·효과음", "무음·반주", "유해 장면 모음", "기타"),
        )
        pipeline = LivePipeline(this)
        choose.setOnClickListener { chooseMedia.launch(arrayOf("audio/*", "video/*")) }
        start.setOnClickListener { startAnalysis() }
        playback.setOnClickListener { requestPlaybackCapture(null) }
        benchmarkButton.setOnClickListener { startBenchmark() }
        stop.setOnClickListener { stopAnalysis("사용자 중지") }
        shareButton.setOnClickListener { shareLastBenchmark() }
        result.text = "모델 구성: ${BuildConfig.MODEL_PROFILE}\n속도 정책: ${WhisperBase.DECODER_POLICY} · ONNX ${OrtTuning.INTRA_OP_THREADS} threads\n\n연속 시험에서는 콘텐츠 종류와 시간을 정한 뒤 다른 앱에서 해당 콘텐츠를 재생하세요."
    }

    private fun startAnalysis() {
        val uri = selected ?: return
        if (!analyzing.compareAndSet(false, true)) return
        setRunningUi(true)
        val target = serverUrl.text.toString().trim().takeIf { it.isNotBlank() }
        worker.execute {
            val report = runCatching {
                val decoder = MediaFileDecoder(this)
                decoder.decodeWindows(uri, { analyzing.get() }) { index, window ->
                    val r = pipeline.process(window, target)
                    runOnUiThread { result.text = formatWindow("미디어", index, r, target, null) }
                }
            }
            runOnUiThread {
                analyzing.set(false)
                setRunningUi(false)
                result.text = report.fold(
                    onSuccess = { summary -> "미디어 분석 완료\n원본: ${summary.sourceRate} Hz · ${summary.sourceChannels}채널\n처리: ${summary.windows}개 4초 구간" },
                    onFailure = { e -> if (e is CancellationException) "미디어 분석을 중지했습니다." else "미디어 분석 실패\n${e.stackTraceToString()}" },
                )
            }
        }
    }

    private fun startBenchmark() {
        val minutes = durationMinutes.text.toString().toIntOrNull()
        if (minutes == null || minutes !in 1..360) {
            result.text = "시험 시간은 1분부터 360분 사이로 입력하세요. 처음 확인할 때는 5분, 본 시험은 60분을 권장합니다."
            return
        }
        val target = serverUrl.text.toString().trim().takeIf { it.isNotBlank() }
        requestPlaybackCapture(BenchmarkConfig(
            contentType = contentType.selectedItem.toString(),
            contentName = contentName.text.toString().trim(),
            durationMinutes = minutes,
            serverEnabled = target != null,
        ))
    }

    private fun requestPlaybackCapture(config: BenchmarkConfig?) {
        if (android.os.Build.VERSION.SDK_INT < android.os.Build.VERSION_CODES.Q) {
            result.text = "재생음 캡처는 Android 10 이상에서 지원됩니다."
            return
        }
        if (analyzing.get()) return
        pendingBenchmark = config
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_AUDIO_PERMISSION)
            return
        }
        val manager = getSystemService(MediaProjectionManager::class.java)
        requestProjection.launch(manager.createScreenCaptureIntent())
    }

    private fun startPlaybackService(resultCode: Int, data: Intent, config: BenchmarkConfig?) {
        if (!analyzing.compareAndSet(false, true)) return
        submittedWindows.set(0)
        completedWindows.set(0)
        benchmark = config?.let { ContinuousBenchmark(this, it) }
        pipeline.onServerMetric = benchmark?.let { active -> { metric -> active.recordServer(metric) } }
        val target = serverUrl.text.toString().trim().takeIf { it.isNotBlank() }

        PlaybackCaptureBus.onWindow = windowCallback@{ window ->
            val active = benchmark
            if (!analyzing.get()) return@windowCallback
            if (active?.expired == true) {
                runOnUiThread { stopAnalysis("계획 시간 완료") }
                return@windowCallback
            }
            val index = submittedWindows.incrementAndGet()
            val capturedAt = active?.elapsedMs ?: SystemClock.elapsedRealtime()
            // The current window is being processed, so only later windows count as backlog.
            val queueDepth = (index - completedWindows.get() - 1).coerceAtLeast(0)
            worker.execute {
                if (!analyzing.get()) return@execute
                val queueStarted = active?.elapsedMs ?: SystemClock.elapsedRealtime()
                val report = runCatching { pipeline.process(window, target) }
                val processed = completedWindows.incrementAndGet()
                report.onSuccess { r ->
                    active?.recordWindow(index, capturedAt, queueStarted - capturedAt, queueDepth, window, r)
                }.onFailure { e ->
                    active?.recordError(index, e.stackTraceToString())
                }
                runOnUiThread {
                    result.text = report.fold(
                        onSuccess = { r -> formatWindow("재생음", index, r, target, active?.progress(processed, submittedWindows.get())) },
                        onFailure = { e -> "재생음 분석 실패\n${e.stackTraceToString()}" },
                    )
                    if (active?.expired == true && analyzing.get()) stopAnalysis("계획 시간 완료")
                }
            }
        }
        setRunningUi(true)
        result.text = if (config == null) {
            "기기 재생음을 듣는 중… 4초마다 분석합니다. 알림창의 재생음 캡처 표시가 유지됩니다."
        } else {
            "연속 시험 시작\n콘텐츠: ${config.contentType}\n콘텐츠명·메모: ${config.contentName.ifBlank { "(미입력)" }}\n계획 시간: ${config.durationMinutes}분\n\n이제 다른 앱에서 콘텐츠를 재생하세요. 매 4초 결과를 즉시 저장합니다."
        }
        ContextCompat.startForegroundService(this, PlaybackCaptureService.intent(this, resultCode, data))
    }

    private fun formatWindow(prefix: String, index: Int, r: LivePipeline.Result, target: String?, progress: String?): String {
        val detail = "%s %d번째 4초 구간 완료\n\nCED %.3f · KoELECTRA %.3f\n%s\n전체 %.1f ms · RTF %.3f\nCED %.1f · Demucs %.1f · Whisper %.1f · KoELECTRA %.1f ms\nWhisper 종료: %s · %d토큰\n\n받아쓰기\n%s\n\n%s".format(
            prefix, index, r.acoustic, r.text,
            if (r.alert) "ALERT — Qwen 전송 대상" else "SAFE",
            r.elapsedMs, r.elapsedMs / 4000.0,
            r.cedMs, r.demucsMs, r.whisperMs, r.koElectraMs,
            r.whisperStopReason, r.tokenCount,
            r.transcript.ifBlank { "(빈 문장)" },
            if (target == null) "서버 주소 없음: 기기 내부 판정만 수행" else "서버 주소 설정됨: ALERT일 때 전송",
        )
        return if (progress == null) detail else "$progress\n$detail"
    }

    private fun stopAnalysis(reason: String) {
        if (!analyzing.getAndSet(false)) return
        stopService(Intent(this, PlaybackCaptureService::class.java))
        PlaybackCaptureBus.onWindow = null
        pipeline.onServerMetric = null
        val finished = benchmark?.finish(reason, submittedWindows.get(), completedWindows.get())
        benchmark = null
        if (finished != null) {
            lastBenchmark = finished
            shareButton.isEnabled = true
            result.text = finished.summaryText
        } else {
            result.text = "미디어 분석을 중지했습니다."
        }
        setRunningUi(false)
    }

    private fun setRunningUi(running: Boolean) {
        choose.isEnabled = !running
        start.isEnabled = !running && selected != null
        playback.isEnabled = !running
        benchmarkButton.isEnabled = !running
        contentType.isEnabled = !running
        contentName.isEnabled = !running
        durationMinutes.isEnabled = !running
        serverUrl.isEnabled = !running
        stop.isEnabled = running
        shareButton.isEnabled = !running && lastBenchmark != null
    }

    private fun shareLastBenchmark() {
        val files = lastBenchmark ?: return
        val uris = arrayListOf(files.summary, files.csv, files.jsonl).mapTo(ArrayList()) { file ->
            FileProvider.getUriForFile(this, "$packageName.files", file)
        }
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND_MULTIPLE).apply {
            type = "text/plain"
            putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
            putExtra(Intent.EXTRA_SUBJECT, "Danger Audio 연속 시험 ${files.directory.name}")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }, "시험 결과 공유"))
    }

    override fun onRequestPermissionsResult(code: Int, permissions: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(code, permissions, results)
        if (code == REQUEST_AUDIO_PERMISSION && results.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            requestPlaybackCapture(pendingBenchmark)
        } else if (code == REQUEST_AUDIO_PERMISSION) {
            pendingBenchmark = null
            result.text = "재생음 캡처에는 오디오 권한이 필요합니다."
        }
    }

    override fun onDestroy() {
        if (analyzing.get()) stopAnalysis("앱 종료")
        pipeline.close()
        worker.shutdown()
        super.onDestroy()
    }

    companion object { private const val REQUEST_AUDIO_PERMISSION = 101 }
}
