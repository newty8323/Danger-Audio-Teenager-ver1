package kr.sht.dangeraudio.onnxbench

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import kotlin.math.sqrt

/** Configuration chosen before an hour-long playback-capture benchmark. */
data class BenchmarkConfig(
    val contentType: String,
    val contentName: String,
    val durationMinutes: Int,
    val serverEnabled: Boolean,
)

data class BenchmarkFiles(
    val directory: File,
    val jsonl: File,
    val csv: File,
    val summary: File,
    val summaryText: String,
)

/**
 * Durable logger for continuous real-time tests.
 *
 * Every processed four-second window is flushed immediately so a thermal
 * shutdown or app crash still leaves all measurements completed before it.
 */
class ContinuousBenchmark(
    private val context: Context,
    val config: BenchmarkConfig,
) : AutoCloseable {
    private val sessionId = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS")
        .withZone(ZoneId.systemDefault()).format(Instant.now())
    private val directory = File(context.filesDir, "benchmarks/$sessionId").apply { mkdirs() }
    private val jsonlFile = File(directory, "windows.jsonl")
    private val csvFile = File(directory, "windows.csv")
    private val summaryFile = File(directory, "summary.txt")
    private val jsonl = BufferedWriter(FileWriter(jsonlFile, false))
    private val csv = BufferedWriter(FileWriter(csvFile, false))
    private val startElapsedMs = SystemClock.elapsedRealtime()
    private val startWallMs = System.currentTimeMillis()
    private val startBattery = resourceSnapshot()
    private val totalTimes = ArrayList<Double>()
    private val rtfValues = ArrayList<Double>()
    private val captureIntervals = ArrayList<Double>()
    private val queueDelays = ArrayList<Double>()
    private val cedTimes = ArrayList<Double>()
    private val demucsTimes = ArrayList<Double>()
    private val whisperTimes = ArrayList<Double>()
    private val whisperMelTimes = ArrayList<Double>()
    private val whisperEncoderTimes = ArrayList<Double>()
    private val whisperDecoderTimes = ArrayList<Double>()
    private val koElectraTimes = ArrayList<Double>()
    private val cpuValues = ArrayList<Double>()
    private val normalizedCpuValues = ArrayList<Double>()
    private val serverTimes = ArrayList<Double>()
    private var alerts = 0
    private var acousticAlerts = 0
    private var textAlerts = 0
    private var emptyTranscripts = 0
    private var repeatedTranscripts = 0
    private var repetitionStops = 0
    private var tokenLimitStops = 0
    private var noSpeechStops = 0
    private var silentInputs = 0
    private var errors = 0
    private var serverDispatched = 0
    private var serverCompleted = 0
    private var serverFailed = 0
    private var maxQueueDepth = 0
    private var maxPssKb = startBattery.pssKb
    private var maxTemperatureC = startBattery.temperatureC
    private var maxThermalStatus = startBattery.thermalStatus
    private var thermalPressureWindows = 0
    private var pluggedWindows = 0
    private var lastCpuMs = Process.getElapsedCpuTime()
    private var lastCpuWallMs = startElapsedMs
    private var lastCaptureElapsedMs = 0L
    private var closed = false

    init {
        jsonl.write(JSONObject().apply {
            put("event_type", "session_start")
            put("session_id", sessionId)
            put("started_at_epoch_ms", startWallMs)
            put("content_type", config.contentType)
            put("content_name", config.contentName)
            put("planned_minutes", config.durationMinutes)
            put("model_profile", BuildConfig.MODEL_PROFILE)
            put("server_enabled", config.serverEnabled)
            put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
            put("android_sdk", Build.VERSION.SDK_INT)
            put("battery_percent", startBattery.batteryPercent)
            put("battery_charge_counter_uah", startBattery.chargeCounterUah)
            put("battery_energy_counter_nwh", startBattery.energyCounterNwh)
            put("battery_plugged", startBattery.plugged)
            put("temperature_c", startBattery.temperatureC)
            put("thermal_status", startBattery.thermalStatus)
            put("available_processors", Runtime.getRuntime().availableProcessors())
            put("onnx_intra_op_threads", OrtTuning.INTRA_OP_THREADS)
            put("onnx_inter_op_threads", OrtTuning.INTER_OP_THREADS)
            put("whisper_decoder_policy", WhisperBase.DECODER_POLICY)
            put("whisper_no_speech_threshold", WhisperBase.NO_SPEECH_THRESHOLD)
            put("whisper_log_probability_threshold", WhisperBase.LOG_PROBABILITY_THRESHOLD)
        }.toString())
        jsonl.newLine(); jsonl.flush()
        csv.write(
            "window_index,capture_elapsed_ms,capture_interval_ms,queue_delay_ms,queue_depth,input_rms," +
                "ced_ms,demucs_ms,whisper_ms,whisper_mel_ms,whisper_encoder_ms," +
                "whisper_decoder_ms,koelectra_ms,total_ms,rtf,input_speech_seconds," +
                "tokens,whisper_stop_reason,whisper_no_speech_probability,whisper_average_log_probability," +
                "acoustic_score,text_score,alert,server_dispatched,empty_transcript," +
                "repeated_transcript,cpu_percent_one_core,cpu_percent_device,pss_kb,java_heap_kb,battery_percent," +
                "battery_current_ua,battery_charge_counter_uah,battery_energy_counter_nwh,battery_plugged," +
                "temperature_c,thermal_status,thermal_pressure,transcript\n"
        )
        csv.flush()
    }

    val elapsedMs: Long get() = SystemClock.elapsedRealtime() - startElapsedMs
    val targetMs: Long get() = config.durationMinutes * 60_000L
    val expired: Boolean get() = elapsedMs >= targetMs

    @Synchronized
    fun recordWindow(
        index: Int,
        captureElapsedMs: Long,
        queueDelayMs: Long,
        queueDepth: Int,
        input: FloatArray,
        result: LivePipeline.Result,
    ) {
        if (closed) return
        val resource = resourceSnapshot()
        val now = SystemClock.elapsedRealtime()
        val cpuNow = Process.getElapsedCpuTime()
        val cpuWall = (now - lastCpuWallMs).coerceAtLeast(1L)
        val cpuPercent = (cpuNow - lastCpuMs) * 100.0 / cpuWall
        val normalizedCpuPercent = cpuPercent / Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        lastCpuMs = cpuNow; lastCpuWallMs = now
        val inputRms = rms(input)
        val empty = result.transcript.isBlank()
        val repeated = isDegenerate(result.transcript)
        val rtf = result.elapsedMs / WINDOW_MS
        val captureIntervalMs = (captureElapsedMs - lastCaptureElapsedMs).toDouble()
        lastCaptureElapsedMs = captureElapsedMs

        totalTimes += result.elapsedMs
        rtfValues += rtf
        captureIntervals += captureIntervalMs
        queueDelays += queueDelayMs.toDouble()
        cedTimes += result.cedMs
        demucsTimes += result.demucsMs
        whisperTimes += result.whisperMs
        whisperMelTimes += result.whisperMelMs
        whisperEncoderTimes += result.whisperEncoderMs
        whisperDecoderTimes += result.whisperDecoderMs
        koElectraTimes += result.koElectraMs
        cpuValues += cpuPercent
        normalizedCpuValues += normalizedCpuPercent
        if (result.alert) alerts++
        if (result.acoustic >= LivePipeline.ACOUSTIC_THRESHOLD) acousticAlerts++
        if (result.text >= LivePipeline.TEXT_THRESHOLD) textAlerts++
        if (empty) emptyTranscripts++
        if (repeated) repeatedTranscripts++
        if (result.whisperStopReason == "repetition") repetitionStops++
        if (result.whisperStopReason == "token_limit") tokenLimitStops++
        if (result.whisperStopReason == "no_speech") noSpeechStops++
        if (inputRms < SILENT_INPUT_RMS) silentInputs++
        if (result.serverDispatched) serverDispatched++
        maxQueueDepth = maxOf(maxQueueDepth, queueDepth)
        maxPssKb = maxOf(maxPssKb, resource.pssKb)
        maxTemperatureC = maxOf(maxTemperatureC, resource.temperatureC)
        maxThermalStatus = maxOf(maxThermalStatus, resource.thermalStatus)
        if (resource.thermalStatus >= PowerManager.THERMAL_STATUS_MODERATE) thermalPressureWindows++
        if (resource.plugged) pluggedWindows++

        jsonl.write(JSONObject().apply {
            put("event_type", "window")
            put("session_id", sessionId)
            put("window_index", index)
            put("capture_elapsed_ms", captureElapsedMs)
            put("capture_interval_ms", captureIntervalMs)
            put("queue_delay_ms", queueDelayMs)
            put("queue_depth", queueDepth)
            put("input_rms", inputRms)
            put("ced_ms", result.cedMs)
            put("demucs_ms", result.demucsMs)
            put("whisper_ms", result.whisperMs)
            put("whisper_mel_ms", result.whisperMelMs)
            put("whisper_encoder_ms", result.whisperEncoderMs)
            put("whisper_decoder_ms", result.whisperDecoderMs)
            put("koelectra_ms", result.koElectraMs)
            put("total_ms", result.elapsedMs)
            put("rtf", rtf)
            put("input_speech_seconds", result.inputSpeechSeconds)
            put("tokens", result.tokenCount)
            put("whisper_stop_reason", result.whisperStopReason)
            put("whisper_no_speech_probability", result.whisperNoSpeechProbability)
            put("whisper_average_log_probability", result.whisperAverageLogProbability)
            put("whisper_warm_session", result.whisperWarmSession)
            put("acoustic_score", result.acoustic)
            put("text_score", result.text)
            put("alert", result.alert)
            put("server_dispatched", result.serverDispatched)
            put("empty_transcript", empty)
            put("repeated_transcript", repeated)
            put("cpu_percent_one_core", cpuPercent)
            put("cpu_percent_device", normalizedCpuPercent)
            put("pss_kb", resource.pssKb)
            put("java_heap_kb", resource.javaHeapKb)
            put("battery_percent", resource.batteryPercent)
            put("battery_current_ua", resource.batteryCurrentUa)
            put("battery_charge_counter_uah", resource.chargeCounterUah)
            put("battery_energy_counter_nwh", resource.energyCounterNwh)
            put("battery_plugged", resource.plugged)
            put("temperature_c", resource.temperatureC)
            put("thermal_status", resource.thermalStatus)
            put("thermal_pressure", resource.thermalStatus >= PowerManager.THERMAL_STATUS_MODERATE)
            put("transcript", result.transcript)
        }.toString())
        jsonl.newLine(); jsonl.flush()

        val values = listOf(
            index, captureElapsedMs, captureIntervalMs, queueDelayMs, queueDepth, inputRms,
            result.cedMs, result.demucsMs, result.whisperMs, result.whisperMelMs,
            result.whisperEncoderMs, result.whisperDecoderMs, result.koElectraMs,
            result.elapsedMs, rtf, result.inputSpeechSeconds, result.tokenCount,
            result.whisperStopReason, result.whisperNoSpeechProbability, result.whisperAverageLogProbability,
            result.acoustic, result.text, result.alert, result.serverDispatched,
            empty, repeated, cpuPercent, normalizedCpuPercent, resource.pssKb, resource.javaHeapKb,
            resource.batteryPercent, resource.batteryCurrentUa, resource.chargeCounterUah,
            resource.energyCounterNwh, resource.plugged, resource.temperatureC,
            resource.thermalStatus, resource.thermalStatus >= PowerManager.THERMAL_STATUS_MODERATE,
            csvEscape(result.transcript),
        )
        csv.write(values.joinToString(",")); csv.newLine(); csv.flush()
    }

    @Synchronized
    fun recordError(index: Int, message: String) {
        if (closed) return
        errors++
        jsonl.write(JSONObject().apply {
            put("event_type", "window_error")
            put("session_id", sessionId)
            put("window_index", index)
            put("elapsed_ms", elapsedMs)
            put("message", message.take(2_000))
        }.toString())
        jsonl.newLine(); jsonl.flush()
    }

    @Synchronized
    fun recordServer(metric: LivePipeline.ServerMetric) {
        if (closed) return
        if (metric.success) serverCompleted++ else serverFailed++
        serverTimes += metric.elapsedMs
        jsonl.write(JSONObject().apply {
            put("event_type", "server")
            put("session_id", sessionId)
            put("request_id", metric.requestId)
            put("elapsed_ms", metric.elapsedMs)
            put("success", metric.success)
            put("http_status", metric.httpStatus)
            put("error", metric.error)
        }.toString())
        jsonl.newLine(); jsonl.flush()
    }

    @Synchronized
    fun progress(processed: Int, submitted: Int): String {
        val minutes = elapsedMs / 60_000.0
        val p95 = percentile(totalTimes, 0.95)
        return "연속 시험: ${config.contentType}\n" +
            "진행 %.1f / %d분 · 처리 %d / 수집 %d창\n".format(minutes, config.durationMinutes, processed, submitted) +
            "현재 p95 %.1f ms · p95 RTF %.3f · 최대 대기 %d창\n".format(
                p95, percentile(rtfValues, 0.95), maxQueueDepth
            ) +
            "ALERT %d · 빈 문장 %d · 반복 환각 %d\n".format(alerts, emptyTranscripts, repeatedTranscripts)
    }

    @Synchronized
    fun finish(reason: String, submitted: Int, processed: Int): BenchmarkFiles {
        if (closed) {
            return BenchmarkFiles(directory, jsonlFile, csvFile, summaryFile, summaryFile.readText())
        }
        val end = resourceSnapshot()
        val dropped = (submitted - processed).coerceAtLeast(0)
        val batteryPercentUsed = if (startBattery.batteryPercent >= 0 && end.batteryPercent >= 0) {
            startBattery.batteryPercent - end.batteryPercent
        } else null
        val chargeUsedMah = counterDelta(startBattery.chargeCounterUah, end.chargeCounterUah)?.div(1_000.0)
        val energyUsedMwh = counterDelta(startBattery.energyCounterNwh, end.energyCounterNwh)?.div(1_000_000.0)
        val driftRatio = performanceDriftRatio(totalTimes)
        val text = buildString {
            appendLine("Danger Audio 연속 시험 결과")
            appendLine("세션: $sessionId")
            appendLine("콘텐츠: ${config.contentType}")
            appendLine("콘텐츠명·메모: ${config.contentName.ifBlank { "(미입력)" }}")
            appendLine("종료 사유: $reason")
            appendLine("모델 프로필: ${BuildConfig.MODEL_PROFILE}")
            appendLine("ONNX 스레드 intra/inter: ${OrtTuning.INTRA_OP_THREADS} / ${OrtTuning.INTER_OP_THREADS}")
            appendLine("Whisper decoder 정책: ${WhisperBase.DECODER_POLICY}")
            appendLine("Whisper 무음 판정 임계값: no-speech > ${WhisperBase.NO_SPEECH_THRESHOLD}, 평균 logP < ${WhisperBase.LOG_PROBABILITY_THRESHOLD}")
            appendLine("실행 시간: %.2f분 / 계획 %d분".format(elapsedMs / 60_000.0, config.durationMinutes))
            appendLine("수집/처리/미처리 창: $submitted / $processed / $dropped")
            appendLine("처리시간 평균/p50/p95/p99: %.1f / %.1f / %.1f / %.1f ms".format(
                average(totalTimes), percentile(totalTimes, .50), percentile(totalTimes, .95), percentile(totalTimes, .99)
            ))
            appendLine("RTF 평균/p95/p99: %.3f / %.3f / %.3f".format(
                average(rtfValues), percentile(rtfValues, .95), percentile(rtfValues, .99)
            ))
            appendLine("단계별 평균/p95 (ms)")
            appendLine("- CED: %.1f / %.1f".format(average(cedTimes), percentile(cedTimes, .95)))
            appendLine("- Demucs: %.1f / %.1f".format(average(demucsTimes), percentile(demucsTimes, .95)))
            appendLine("- Whisper 전체: %.1f / %.1f".format(average(whisperTimes), percentile(whisperTimes, .95)))
            appendLine("  - log-Mel: %.1f / %.1f".format(average(whisperMelTimes), percentile(whisperMelTimes, .95)))
            appendLine("  - encoder: %.1f / %.1f".format(average(whisperEncoderTimes), percentile(whisperEncoderTimes, .95)))
            appendLine("  - decoder: %.1f / %.1f".format(average(whisperDecoderTimes), percentile(whisperDecoderTimes, .95)))
            appendLine("- KoELECTRA: %.1f / %.1f".format(average(koElectraTimes), percentile(koElectraTimes, .95)))
            appendLine("캡처 간격 평균/p95: %.1f / %.1f ms".format(
                average(captureIntervals), percentile(captureIntervals, .95)
            ))
            appendLine("대기시간 평균/p95: %.1f / %.1f ms".format(
                average(queueDelays), percentile(queueDelays, .95)
            ))
            appendLine("프로세스 CPU 평균/p95: %.1f / %.1f %%".format(
                average(cpuValues), percentile(cpuValues, .95)
            ))
            appendLine("기기 전체 CPU 용량 대비 평균/p95: %.1f / %.1f %%".format(
                average(normalizedCpuValues), percentile(normalizedCpuValues, .95)
            ))
            appendLine("후반/초반 처리시간 비율: %.3f배".format(driftRatio))
            appendLine("최대 대기 깊이: $maxQueueDepth")
            appendLine("ALERT/음향/텍스트: $alerts / $acousticAlerts / $textAlerts")
            appendLine("빈 문장/반복 환각/저음량 입력: $emptyTranscripts / $repeatedTranscripts / $silentInputs")
            appendLine("Whisper no-speech/반복 조기종료/토큰 상한 종료: $noSpeechStops / $repetitionStops / $tokenLimitStops")
            appendLine("오류: $errors")
            val serverPending = (serverDispatched - serverCompleted - serverFailed).coerceAtLeast(0)
            appendLine("서버 전송/성공/실패/대기: $serverDispatched / $serverCompleted / $serverFailed / $serverPending")
            appendLine("서버 응답시간 평균/p95: %.1f / %.1f ms".format(
                average(serverTimes), percentile(serverTimes, .95)
            ))
            appendLine("PSS 시작/최대/종료: ${startBattery.pssKb} / $maxPssKb / ${end.pssKb} KB")
            appendLine("배터리 시작/종료: ${startBattery.batteryPercent}% / ${end.batteryPercent}%")
            appendLine("배터리 잔량 소모: ${batteryPercentUsed?.let { "$it%p" } ?: "측정 불가"}")
            appendLine("배터리 충전량 소모: ${chargeUsedMah?.let { "%.2f mAh".format(it) } ?: "측정 불가"}")
            appendLine("배터리 에너지 소모: ${energyUsedMwh?.let { "%.2f mWh".format(it) } ?: "측정 불가"}")
            appendLine("충전 연결 상태 시작/종료·연결 표본: ${startBattery.plugged} / ${end.plugged} · $pluggedWindows")
            appendLine("온도 시작/최대/종료: %.1f / %.1f / %.1f ℃".format(
                startBattery.temperatureC, maxTemperatureC, end.temperatureC
            ))
            appendLine("thermal 시작/최대/종료: ${startBattery.thermalStatus} / $maxThermalStatus / ${end.thermalStatus}")
            appendLine("thermal MODERATE 이상 구간: $thermalPressureWindows")
            appendLine()
            appendLine("실시간 1차 판정")
            appendLine("- p95 RTF < 1: ${percentile(rtfValues, .95) < 1.0}")
            appendLine("- p95 처리시간 < 4000 ms: ${percentile(totalTimes, .95) < WINDOW_MS}")
            appendLine("- 미처리 창 0: ${dropped == 0}")
            appendLine("- 오류 0: ${errors == 0}")
            appendLine()
            appendLine("파일: ${directory.absolutePath}")
        }
        summaryFile.writeText(text)
        jsonl.write(JSONObject().apply {
            put("event_type", "session_end")
            put("session_id", sessionId)
            put("reason", reason)
            put("submitted", submitted)
            put("processed", processed)
            put("dropped", dropped)
            put("p95_total_ms", percentile(totalTimes, .95))
            put("p95_rtf", percentile(rtfValues, .95))
            put("max_queue_depth", maxQueueDepth)
            put("errors", errors)
            put("battery_percent_used", batteryPercentUsed ?: JSONObject.NULL)
            put("battery_charge_used_mah", chargeUsedMah ?: JSONObject.NULL)
            put("battery_energy_used_mwh", energyUsedMwh ?: JSONObject.NULL)
            put("max_temperature_c", maxTemperatureC)
            put("max_thermal_status", maxThermalStatus)
            put("thermal_pressure_windows", thermalPressureWindows)
            put("performance_drift_ratio", driftRatio)
        }.toString())
        jsonl.newLine(); jsonl.flush(); csv.flush()
        jsonl.close(); csv.close(); closed = true
        return BenchmarkFiles(directory, jsonlFile, csvFile, summaryFile, text)
    }

    override fun close() {
        if (!closed) finish("앱 종료", 0, 0)
    }

    private data class ResourceSnapshot(
        val pssKb: Long,
        val javaHeapKb: Long,
        val batteryPercent: Int,
        val batteryCurrentUa: Int,
        val chargeCounterUah: Int,
        val energyCounterNwh: Long,
        val plugged: Boolean,
        val temperatureC: Double,
        val thermalStatus: Int,
    )

    private fun resourceSnapshot(): ResourceSnapshot {
        val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = battery?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val percent = if (level < 0 || scale <= 0) -1 else level * 100 / scale
        val rawTemperature = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
        val manager = context.getSystemService(BatteryManager::class.java)
        val current = manager?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW) ?: Int.MIN_VALUE
        val charge = manager?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER) ?: Int.MIN_VALUE
        val energy = manager?.getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER) ?: Long.MIN_VALUE
        val plugged = (battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
        val thermal = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            context.getSystemService(PowerManager::class.java)?.currentThermalStatus ?: -1
        } else -1
        val runtime = Runtime.getRuntime()
        return ResourceSnapshot(
            Debug.getPss(),
            (runtime.totalMemory() - runtime.freeMemory()) / 1024,
            percent,
            current,
            charge,
            energy,
            plugged,
            rawTemperature / 10.0,
            thermal,
        )
    }

    private fun rms(values: FloatArray): Double {
        if (values.isEmpty()) return 0.0
        var sum = 0.0
        values.forEach { sum += it * it }
        return sqrt(sum / values.size)
    }

    private fun isDegenerate(text: String): Boolean {
        val compact = text.replace(Regex("[\\s.,!?~…]+"), "")
        if (compact.length < 8) return false
        for (unit in 1..minOf(6, compact.length / 4)) {
            val prefix = compact.substring(0, unit)
            if (compact.chunked(unit).count { it == prefix } >= 4 && compact.startsWith(prefix.repeat(4))) return true
        }
        val words = text.split(Regex("\\s+")).filter { it.isNotBlank() }
        return words.size >= 6 && words.distinct().size.toDouble() / words.size <= 0.35
    }

    private fun percentile(values: List<Double>, quantile: Double): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val index = kotlin.math.ceil(quantile * sorted.size).toInt().coerceIn(1, sorted.size) - 1
        return sorted[index]
    }

    private fun average(values: List<Double>) = if (values.isEmpty()) 0.0 else values.average()
    private fun counterDelta(start: Int, end: Int): Double? =
        if (start == Int.MIN_VALUE || end == Int.MIN_VALUE) null else (start - end).toDouble()

    private fun counterDelta(start: Long, end: Long): Double? =
        if (start == Long.MIN_VALUE || end == Long.MIN_VALUE) null else (start - end).toDouble()

    /** Compares warm early windows with the same number of final windows. */
    private fun performanceDriftRatio(values: List<Double>): Double {
        if (values.size < 6) return 0.0
        val count = minOf(10, (values.size - 1) / 2)
        val early = values.subList(1, 1 + count).average() // Skip cold model/session initialization.
        val late = values.takeLast(count).average()
        return if (early <= 0.0) 0.0 else late / early
    }

    private fun csvEscape(value: String) = "\"${value.replace("\"", "\"\"").replace("\n", " ")}\""

    companion object {
        private const val WINDOW_MS = 4_000.0
        private const val SILENT_INPUT_RMS = 0.0005
    }
}
