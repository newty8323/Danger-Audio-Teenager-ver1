package kr.sht.dangeraudio.onnxbench

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var result: TextView; private lateinit var serverUrl: EditText
    private lateinit var start: Button; private lateinit var stop: Button; private lateinit var pipeline: LivePipeline
    private var capture: LiveAudioCapture? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState); setContentView(R.layout.activity_main)
        result = findViewById(R.id.result); serverUrl = findViewById(R.id.serverUrl); start = findViewById(R.id.startButton); stop = findViewById(R.id.stopButton)
        pipeline = LivePipeline(this); start.setOnClickListener { startCapture() }; stop.setOnClickListener { stopCapture() }
    }
    private fun startCapture() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) { ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_MIC); return }
        if (capture != null) return
        result.text = "마이크를 듣는 중… 첫 4초 구간부터 CED → Demucs → Whisper → KoELECTRA를 실행합니다."
        start.isEnabled = false; stop.isEnabled = true
        capture = LiveAudioCapture(this) { window ->
            val target = serverUrl.text.toString().trim().takeIf { it.isNotBlank() }
            worker.execute {
                val report = runCatching { pipeline.process(window, target) }
                runOnUiThread {
                    result.text = report.fold(
                        onSuccess = { r -> "실시간 4초 분석 완료\n\nCED-mini 음향 위험도: %.3f\nKoELECTRA 언어 위험도: %.3f\n최종 상태: %s\n처리시간: %.1f ms\nRTF: %.3f\n\n받아쓰기\n%s\n\n%s".format(r.acoustic, r.text, if (r.alert) "ALERT — Qwen 전송 대상" else "SAFE", r.elapsedMs, r.elapsedMs / 4000.0, r.transcript.ifBlank { "(빈 문장)" }, if (target == null) "서버 주소 없음: 기기 내부 판정만 수행" else "서버 주소 설정됨: ALERT일 때 음성·텍스트 전송") },
                        onFailure = { e -> "실시간 분석 실패\n${e.stackTraceToString()}" },
                    )
                }
            }
        }.also { it.start() }
    }
    private fun stopCapture() { capture?.close(); capture = null; start.isEnabled = true; stop.isEnabled = false; result.text = "중지됨. 다시 시작하면 새 4초 마이크 구간부터 분석합니다." }
    override fun onRequestPermissionsResult(code: Int, permissions: Array<out String>, results: IntArray) { super.onRequestPermissionsResult(code, permissions, results); if (code == REQUEST_MIC && results.firstOrNull() == PackageManager.PERMISSION_GRANTED) startCapture() else if (code == REQUEST_MIC) result.text = "마이크 권한이 필요합니다." }
    override fun onDestroy() { stopCapture(); pipeline.close(); worker.shutdown(); super.onDestroy() }
    companion object { private const val REQUEST_MIC = 100 }
}
