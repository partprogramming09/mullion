// Times a native 32-bit atomic add against a 64-bit add emulated with a lock, as Mullion builds it for GPUs without
// 64-bit atomics, at several contention levels. Usage: metal_atomic64_cost [threads] [repeats]
import Metal

let threads = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1])! : 1 << 20
let repeats = CommandLine.arguments.count > 2 ? Int(CommandLine.arguments[2])! : 5
let source = """
#include <metal_stdlib>
using namespace metal;

kernel void native32(device atomic_uint *words [[buffer(0)]], constant uint &spread [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
  atomic_fetch_add_explicit(words + id % spread, 1u, memory_order_relaxed);
}

// one lane of a SIMD group at a time takes the word's lock, reads both halves, adds, writes them, and lets it go
kernel void locked64(device atomic_uint *words [[buffer(0)]], device atomic_uint *locks [[buffer(1)]],
                     constant uint &spread [[buffer(2)]], uint id [[thread_position_in_grid]]) {
  uint w = id % spread;
  device atomic_uint *lo = words + 2 * w, *hi = lo + 1;
  device atomic_uint *lock = locks + (w & 0xffffu);
  bool pending = true;
  while (simd_any(pending)) {
    if (pending && simd_is_first()) {
      for (uint spins = 0; spins < (1u << 24); spins++) {
        uint expected = 0;
        if (atomic_compare_exchange_weak_explicit(lock, &expected, 1u, memory_order_relaxed, memory_order_relaxed)) {
          atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
          ulong v = ulong(atomic_fetch_or_explicit(hi, 0u, memory_order_relaxed)) << 32 |
                    atomic_fetch_or_explicit(lo, 0u, memory_order_relaxed);
          v += 1;
          atomic_store_explicit(lo, uint(v), memory_order_relaxed);
          atomic_store_explicit(hi, uint(v >> 32), memory_order_relaxed);
          atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
          atomic_exchange_explicit(lock, 0u, memory_order_relaxed);
          break;
        }
      }
      pending = false;
    }
  }
}
"""

let device = MTLCreateSystemDefaultDevice()!
let library = try device.makeLibrary(source: source, options: nil)
let queue = device.makeCommandQueue()!
let words = device.makeBuffer(length: threads * 8, options: .storageModeShared)!
let locks = device.makeBuffer(length: 65536 * 4, options: .storageModeShared)!

/// Runs `kernel` over `threads` threads spread on `spread` words; returns the best GPU time in ms and the words' sum.
func time(_ kernel: String, spread: Int) throws -> (Double, UInt64) {
  let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: kernel)!)
  var best = Double.infinity
  var sum: UInt64 = 0
  for _ in 0..<repeats {
    memset(words.contents(), 0, words.length)
    memset(locks.contents(), 0, locks.length)
    var s = UInt32(spread)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(words, offset: 0, index: 0)
    enc.setBuffer(locks, offset: 0, index: 1)
    enc.setBytes(&s, length: 4, index: 2)
    enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    best = min(best, (cb.gpuEndTime - cb.gpuStartTime) * 1000)
    if kernel == "native32" {
      sum = (0..<spread).reduce(0) { $0 + UInt64(words.contents().load(fromByteOffset: $1 * 4, as: UInt32.self)) }
    } else {
      sum = (0..<spread).reduce(0) { $0 + words.contents().load(fromByteOffset: $1 * 8, as: UInt64.self) }
    }
  }
  return (best, sum)
}

print("device: \(device.name), \(threads) threads, best of \(repeats)")
print("words      native32 ms   locked64 ms   ratio   sums ok")
for spread in [threads, 65536, 1024, 64, 1] {
  let (n, ns) = try time("native32", spread: spread)
  let (l, ls) = try time("locked64", spread: spread)
  let ok = ns == UInt64(threads) && ls == UInt64(threads)
  print(String(format: "%-9d  %10.3f   %11.3f   %5.1fx   %@", spread, n, l, l / n, ok ? "yes" : "NO"))
}
