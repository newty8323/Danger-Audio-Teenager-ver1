package kr.sht.dangeraudio.onnxbench

import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.activity.result.contract.ActivityResultContracts
import java.util.concurrent.CancellationException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var result: TextView; private lateinit var serverUrl: EditText
    private lateinit var choose: Button; private lateinit var start: Button; private lateinit var stop: Button; private lateinit var pipeline: LivePipeline
    private var selected: Uri? = null
    private val analyzing = AtomicBoolean(false)
    private val chooseMedia = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        contentResolver.takePersistableUriPermission(uri, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
        selected = uri; start.isEnabled = true
        result.text = "선택한 미디어\n${uri.lastPathSegment ?: uri}\n\n분석 시작을 누르면 원본 파일을 16 kHz mono·4초 창으로 직접 처리합니다."
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState); setContentView(R.layout.activity_main)
        result = findViewById(R.id.result); serverUrl = findViewById(R.id.serverUrl); choose = findViewById(R.id.chooseButton); start = findViewById(R.id.startButton); stop = findViewById(R.id.stopButton)
        pipeline = LivePipeline(this); choose.setOnClickListener { chooseMedia.launch(arrayOf("audio/*", "video/*")) }; start.setOnClickListener { startAnalysis() }; stop.setOnClickListener { stopAnalysis() }
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
    private fun stopAnalysis() { analyzing.set(false); result.text = "현재 구간이 끝난 뒤 미디어 분석을 중지합니다." }
    override fun onDestroy() { analyzing.set(false); pipeline.close(); worker.shutdown(); super.onDestroy() }
}
