// Metal capability + performance probe for (possibly virtualized) Apple silicon Macs.
// Build: swiftc -swift-version 5 -O probe.swift -o probe && ./probe
// Prints a human-readable report and a final JSON line prefixed with "RESULT_JSON ".

import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate

var results: [String: Any] = [:]

func log(_ s: String) { print(s); fflush(stdout) }

/// JSONSerialization rejects NaN/inf (e.g. if a VM reports zero GPU timestamps); stringify them.
func sanitize(_ v: Any) -> Any {
    if let d = v as? [String: Any] { return d.mapValues(sanitize) }
    if let a = v as? [Any] { return a.map(sanitize) }
    if let x = v as? Double, !x.isFinite { return "\(x)" }
    return v
}
func printJSON() {
    if let d = try? JSONSerialization.data(withJSONObject: sanitize(results), options: [.sortedKeys]),
       let s = String(data: d, encoding: .utf8) { log("\nRESULT_JSON " + s) }
    else { log("\nRESULT_JSON serialization failed") }
}

// MARK: - System info

func sysctlString(_ name: String) -> String? {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
    return String(cString: buf)
}

func sysctlInt(_ name: String) -> Int64? {
    var v: Int64 = 0
    var size = MemoryLayout<Int64>.size
    guard sysctlbyname(name, &v, &size, nil, 0) == 0 else { return nil }
    return v  // works for 4-byte values too (little-endian, zero-initialized)
}

var sys: [String: Any] = [:]
sys["os"] = ProcessInfo.processInfo.operatingSystemVersionString
for k in ["machdep.cpu.brand_string", "hw.model"] { sys[k] = sysctlString(k) ?? "n/a" }
for k in ["hw.ncpu", "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu", "hw.memsize", "kern.hv_vmm_present"] {
    if let v = sysctlInt(k) { sys[k] = v } else { sys[k] = "n/a" }
}
results["system"] = sys
log("== System")
for (k, v) in sys.sorted(by: { $0.key < $1.key }) { log("  \(k): \(v)") }
log("  (kern.hv_vmm_present = 1 means we are inside a VM)")

// MARK: - Device info

let all = MTLCopyAllDevices()
log("\n== Metal devices: \(all.count) [\(all.map { $0.name }.joined(separator: ", "))]")
guard let device = MTLCreateSystemDefaultDevice() else {
    log("NO DEFAULT METAL DEVICE")
    results["metal_device"] = NSNull()
    printJSON()
    exit(1)
}

var dev: [String: Any] = [:]
dev["name"] = device.name
if #available(macOS 14.0, *) { dev["architecture"] = device.architecture.name }
dev["hasUnifiedMemory"] = device.hasUnifiedMemory
dev["isLowPower"] = device.isLowPower
dev["isHeadless"] = device.isHeadless
dev["recommendedMaxWorkingSetSize_MB"] = Int(device.recommendedMaxWorkingSetSize >> 20)
dev["maxBufferLength_MB"] = device.maxBufferLength >> 20
dev["maxThreadgroupMemoryLength"] = device.maxThreadgroupMemoryLength
let mt = device.maxThreadsPerThreadgroup
dev["maxThreadsPerThreadgroup"] = "\(mt.width)x\(mt.height)x\(mt.depth)"
dev["supportsRaytracing"] = device.supportsRaytracing
dev["supports32BitFloatFiltering"] = device.supports32BitFloatFiltering
dev["supportsBCTextureCompression"] = device.supportsBCTextureCompression

// GPU families via raw values so this compiles on any recent SDK.
var fams: [String] = []
for raw in 1001...1010 { if let f = MTLGPUFamily(rawValue: raw), device.supportsFamily(f) { fams.append("apple\(raw - 1000)") } }
for raw in 2001...2002 { if let f = MTLGPUFamily(rawValue: raw), device.supportsFamily(f) { fams.append("mac\(raw - 2000)") } }
for raw in 3001...3003 { if let f = MTLGPUFamily(rawValue: raw), device.supportsFamily(f) { fams.append("common\(raw - 3000)") } }
for raw in 5001...5002 { if let f = MTLGPUFamily(rawValue: raw), device.supportsFamily(f) { fams.append("metal\(raw - 4998)") } }
dev["families"] = fams

// Performance counters (needed for real profiling).
dev["counterSets"] = (device.counterSets ?? []).map { $0.name }
dev["counterSampling_stageBoundary"] = device.supportsCounterSampling(.atStageBoundary)
dev["counterSampling_dispatchBoundary"] = device.supportsCounterSampling(.atDispatchBoundary)
dev["MPS_supported"] = MPSSupportsMTLDevice(device)
results["device"] = dev
for (k, v) in dev.sorted(by: { $0.key < $1.key }) { log("  \(k): \(v)") }

// MARK: - Kernels

let src = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void vadd(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                 device float* c [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    c[i] = a[i] + b[i];
}

kernel void copy4(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
    dst[i] = src[i];
}

// 8 independent float4 FMA chains per thread, unrolled 4x (loop overhead otherwise caps it at ~40% of peak);
// 8 * 4 * 4 * 2 = 256 flops per iteration.
#define FMA8 x0 = fma(x0, b, c); x1 = fma(x1, b, c); x2 = fma(x2, b, c); x3 = fma(x3, b, c); \
             x4 = fma(x4, b, c); x5 = fma(x5, b, c); x6 = fma(x6, b, c); x7 = fma(x7, b, c);
kernel void fma_loop(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    const float4 b = 0.9999f, c = 0.0001f;
    float4 x0 = float(i) * 1e-9f + float4(0, 1, 2, 3) * 0.01f, x1 = x0 + 0.1f, x2 = x0 + 0.2f, x3 = x0 + 0.3f;
    float4 x4 = x0 + 0.4f, x5 = x0 + 0.5f, x6 = x0 + 0.6f, x7 = x0 + 0.7f;
    for (uint k = 0; k < iters; k++) { FMA8 FMA8 FMA8 FMA8 }
    float4 s = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
    out[i] = s.x + s.y + s.z + s.w;
}

// simdgroup_matrix 8x8 MMA throughput (the primitive metal-flash-attention etc. rely on).
// 4 simdgroups per threadgroup, 4 independent accumulators each; 1024 flops per MMA.
kernel void simd_mma(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    simdgroup_float8x8 a = make_filled_simdgroup_matrix<float, 8, 8>(0.5f);
    simdgroup_float8x8 b = make_filled_simdgroup_matrix<float, 8, 8>(0.999f);
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 c1 = c0, c2 = c0, c3 = c0;
    for (uint k = 0; k < iters; k++) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
    }
    device float* o = out + (tg * 4 + sg) * 4 * 64;
    simdgroup_store(c0, o, 8);
    simdgroup_store(c1, o + 64, 8);
    simdgroup_store(c2, o + 128, 8);
    simdgroup_store(c3, o + 192, 8);
}
"""

let queue = device.makeCommandQueue()!
var tests: [String: Any] = [:]

func pipeline(_ lib: MTLLibrary, _ name: String) -> MTLComputePipelineState? {
    do { return try device.makeComputePipelineState(function: lib.makeFunction(name: name)!) }
    catch { log("  pipeline \(name) failed: \(error)"); return nil }
}

/// Runs `encode` `reps` times (one command buffer each), returns best GPU time and best wall time in seconds.
func timeGPU(_ reps: Int, _ encode: (MTLComputeCommandEncoder) -> Void) -> (gpu: Double, wall: Double, error: String?) {
    var bestGPU = Double.infinity, bestWall = Double.infinity
    for _ in 0..<reps {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        encode(enc)
        enc.endEncoding()
        let t0 = DispatchTime.now().uptimeNanoseconds
        cb.commit()
        cb.waitUntilCompleted()
        let wall = Double(DispatchTime.now().uptimeNanoseconds - t0) * 1e-9
        if let e = cb.error { return (.nan, .nan, "\(e)") }
        bestGPU = min(bestGPU, cb.gpuEndTime - cb.gpuStartTime)
        bestWall = min(bestWall, wall)
    }
    return (bestGPU, bestWall, nil)
}

func fmt(_ x: Double) -> String { String(format: "%.2f", x) }

var lib: MTLLibrary? = nil
do {
    lib = try device.makeLibrary(source: src, options: nil)
    tests["shader_compile"] = "ok"
} catch {
    tests["shader_compile"] = "FAILED: \(error)"
    log("\n== Shader compile FAILED: \(error)")
}

if let lib = lib {
    // Correctness: vadd
    log("\n== vadd correctness")
    if let p = pipeline(lib, "vadd") {
        let n = 1 << 20
        let a = device.makeBuffer(length: n * 4, options: .storageModeShared)!
        let b = device.makeBuffer(length: n * 4, options: .storageModeShared)!
        let c = device.makeBuffer(length: n * 4, options: .storageModeShared)!
        let pa = a.contents().bindMemory(to: Float.self, capacity: n)
        let pb = b.contents().bindMemory(to: Float.self, capacity: n)
        let pc = c.contents().bindMemory(to: Float.self, capacity: n)
        for i in 0..<n { pa[i] = Float(i); pb[i] = 2 * Float(i); pc[i] = -1 }
        let r = timeGPU(1) { e in
            e.setComputePipelineState(p); e.setBuffer(a, offset: 0, index: 0)
            e.setBuffer(b, offset: 0, index: 1); e.setBuffer(c, offset: 0, index: 2)
            e.dispatchThreadgroups(MTLSize(width: n / 256, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        var bad = 0
        for i in 0..<n where pc[i] != 3 * Float(i) { bad += 1 }
        let status = r.error ?? (bad == 0 ? "ok" : "WRONG (\(bad) mismatches)")
        tests["vadd"] = status
        log("  \(status)")
    }

    // Memory bandwidth: copy 256 MB -> 256 MB
    log("\n== Bandwidth (float4 copy, 256 MB)")
    if let p = pipeline(lib, "copy4") {
        let bytes = 256 << 20
        let s = device.makeBuffer(length: bytes, options: .storageModePrivate)!
        let d = device.makeBuffer(length: bytes, options: .storageModePrivate)!
        let n4 = bytes / 16
        let r = timeGPU(10) { e in
            e.setComputePipelineState(p); e.setBuffer(s, offset: 0, index: 0); e.setBuffer(d, offset: 0, index: 1)
            e.dispatchThreadgroups(MTLSize(width: n4 / 256, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        if let err = r.error { tests["bandwidth"] = "FAILED: \(err)"; log("  FAILED: \(err)") }
        else {
            let gbs = 2 * Double(bytes) / r.gpu / 1e9, gbsWall = 2 * Double(bytes) / r.wall / 1e9
            tests["bandwidth_GBps_gpu"] = gbs; tests["bandwidth_GBps_wall"] = gbsWall
            log("  \(fmt(gbs)) GB/s (GPU timestamps), \(fmt(gbsWall)) GB/s (wall)")
        }
    }

    // FP32 FMA throughput
    log("\n== FP32 FMA throughput")
    if let p = pipeline(lib, "fma_loop") {
        let threads = 1 << 20
        var iters: UInt32 = 256
        let out = device.makeBuffer(length: threads * 4, options: .storageModePrivate)!
        let r = timeGPU(5) { e in
            e.setComputePipelineState(p); e.setBuffer(out, offset: 0, index: 0)
            e.setBytes(&iters, length: 4, index: 1)
            e.dispatchThreadgroups(MTLSize(width: threads / 256, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        if let err = r.error { tests["fp32_fma"] = "FAILED: \(err)"; log("  FAILED: \(err)") }
        else {
            let flops = Double(threads) * Double(iters) * 256
            tests["fp32_fma_TFLOPS_gpu"] = flops / r.gpu / 1e12; tests["fp32_fma_TFLOPS_wall"] = flops / r.wall / 1e12
            log("  \(fmt(flops / r.gpu / 1e12)) TFLOPS (GPU timestamps), \(fmt(flops / r.wall / 1e12)) TFLOPS (wall)")
        }
    }

    // simdgroup_matrix MMA throughput + correctness
    log("\n== simdgroup_matrix 8x8 fp32 MMA")
    if let p = pipeline(lib, "simd_mma") {
        log("  threadExecutionWidth: \(p.threadExecutionWidth), maxTotalThreadsPerThreadgroup: \(p.maxTotalThreadsPerThreadgroup)")
        let tgs = 1024
        var iters: UInt32 = 2048
        let out = device.makeBuffer(length: tgs * 4 * 4 * 64 * 4, options: .storageModeShared)!
        let w = p.threadExecutionWidth
        let r = timeGPU(5) { e in
            e.setComputePipelineState(p); e.setBuffer(out, offset: 0, index: 0)
            e.setBytes(&iters, length: 4, index: 1)
            e.dispatchThreadgroups(MTLSize(width: tgs, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 4 * w, height: 1, depth: 1))
        }
        if let err = r.error { tests["simd_mma"] = "FAILED: \(err)"; log("  FAILED: \(err)") }
        else {
            let expected = Float(iters) * 0.5 * 0.999 * 8
            let got = out.contents().bindMemory(to: Float.self, capacity: 1)[0]
            let ok = abs(got - expected) / expected < 1e-3
            let flops = Double(tgs) * 4 * 4 * Double(iters) * 1024
            tests["simd_mma_correct"] = ok ? "ok" : "WRONG (got \(got), expected \(expected))"
            tests["simd_mma_TFLOPS_gpu"] = flops / r.gpu / 1e12; tests["simd_mma_TFLOPS_wall"] = flops / r.wall / 1e12
            log("  correctness: \(tests["simd_mma_correct"]!)")
            log("  \(fmt(flops / r.gpu / 1e12)) TFLOPS (GPU timestamps), \(fmt(flops / r.wall / 1e12)) TFLOPS (wall)")
        }
    }
}

// MARK: - MPS sgemm

let N = 4096
func fillA(_ i: Int) -> Float { Float(i % 7) / 7 - 0.4 }
func fillB(_ i: Int) -> Float { Float(i % 5) / 5 - 0.3 }

log("\n== MPS sgemm \(N)x\(N)")
if MPSSupportsMTLDevice(device) {
    let rowBytes = N * 4
    let A = device.makeBuffer(length: N * rowBytes, options: .storageModeShared)!
    let B = device.makeBuffer(length: N * rowBytes, options: .storageModeShared)!
    let C = device.makeBuffer(length: N * rowBytes, options: .storageModeShared)!
    let pA = A.contents().bindMemory(to: Float.self, capacity: N * N)
    let pB = B.contents().bindMemory(to: Float.self, capacity: N * N)
    for i in 0..<(N * N) { pA[i] = fillA(i); pB[i] = fillB(i) }
    let desc = MPSMatrixDescriptor(rows: N, columns: N, rowBytes: rowBytes, dataType: .float32)
    let mA = MPSMatrix(buffer: A, descriptor: desc), mB = MPSMatrix(buffer: B, descriptor: desc), mC = MPSMatrix(buffer: C, descriptor: desc)
    let gemm = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false,
                                       resultRows: N, resultColumns: N, interiorColumns: N, alpha: 1, beta: 0)
    var best = Double.infinity, bestWall = Double.infinity
    var err: String? = nil
    for _ in 0..<5 {
        let cb = queue.makeCommandBuffer()!
        gemm.encode(commandBuffer: cb, leftMatrix: mA, rightMatrix: mB, resultMatrix: mC)
        let t0 = DispatchTime.now().uptimeNanoseconds
        cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { err = "\(e)"; break }
        best = min(best, cb.gpuEndTime - cb.gpuStartTime)
        bestWall = min(bestWall, Double(DispatchTime.now().uptimeNanoseconds - t0) * 1e-9)
    }
    if let err = err { tests["mps_sgemm"] = "FAILED: \(err)"; log("  FAILED: \(err)") }
    else {
        // Check two entries against a double-precision CPU reference.
        let pC = C.contents().bindMemory(to: Float.self, capacity: N * N)
        var maxRel = 0.0
        for (r, c) in [(0, 0), (N - 1, N - 1), (N / 3, N / 2)] {
            var ref = 0.0
            for k in 0..<N { ref += Double(fillA(r * N + k)) * Double(fillB(k * N + c)) }
            maxRel = max(maxRel, abs(Double(pC[r * N + c]) - ref) / max(abs(ref), 1e-3))
        }
        let flops = 2 * Double(N) * Double(N) * Double(N)
        tests["mps_sgemm_maxRelErr"] = maxRel
        tests["mps_sgemm_TFLOPS_gpu"] = flops / best / 1e12; tests["mps_sgemm_TFLOPS_wall"] = flops / bestWall / 1e12
        log("  max rel err vs CPU: \(String(format: "%.2e", maxRel))")
        log("  \(fmt(flops / best / 1e12)) TFLOPS (GPU timestamps), \(fmt(flops / bestWall / 1e12)) TFLOPS (wall)")
    }
} else {
    tests["mps_sgemm"] = "MPS not supported on this device"
    log("  MPS not supported on this device")
}

// MARK: - CPU sgemm via Accelerate (AMX/SME path)

log("\n== CPU Accelerate sgemm \(N)x\(N)")
do {
    var a = [Float](repeating: 0, count: N * N), b = a, c = a
    for i in 0..<(N * N) { a[i] = fillA(i); b[i] = fillB(i) }
    var best = Double.infinity
    for _ in 0..<3 {
        let t0 = DispatchTime.now().uptimeNanoseconds
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(N), Int32(N), Int32(N), 1, a, Int32(N), b, Int32(N), 0, &c, Int32(N))
        best = min(best, Double(DispatchTime.now().uptimeNanoseconds - t0) * 1e-9)
    }
    let tf = 2 * Double(N) * Double(N) * Double(N) / best / 1e12
    tests["cpu_accelerate_sgemm_TFLOPS"] = tf
    log("  \(fmt(tf)) TFLOPS")
}

results["tests"] = tests
printJSON()
