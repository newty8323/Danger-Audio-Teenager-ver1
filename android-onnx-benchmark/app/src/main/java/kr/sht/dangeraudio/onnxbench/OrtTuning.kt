package kr.sht.dangeraudio.onnxbench

import ai.onnxruntime.OrtSession

/**
 * Shared long-running CPU policy for every ONNX session.
 *
 * ORT's automatic pool can keep many worker threads spinning between four-second
 * windows. On the physical-phone one-hour run this used about 3.7 cores and put
 * Android in MODERATE thermal pressure after eight minutes. Two intra-op
 * threads deliberately trade some latency headroom for lower sustained CPU
 * load and heat; disabling spinning lets those workers sleep between runs.
 */
object OrtTuning {
    const val INTRA_OP_THREADS = 2
    const val INTER_OP_THREADS = 1

    fun createOptions(): OrtSession.SessionOptions = OrtSession.SessionOptions().apply {
        setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setIntraOpNumThreads(INTRA_OP_THREADS)
        setInterOpNumThreads(INTER_OP_THREADS)
        addConfigEntry("session.intra_op.allow_spinning", "0")
        addConfigEntry("session.inter_op.allow_spinning", "0")
    }
}
