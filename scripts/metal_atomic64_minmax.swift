// Compiles a compute kernel with a native 64-bit atomic max on a buffer (or texture) and reports whether the device's
// shader compiler accepts it. Usage: metal_atomic64_minmax [buffer|texture]
import Metal

let mode = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "buffer"
let device = MTLCreateSystemDefaultDevice()!
print("device: \(device.name)")
for (name, family) in [("apple7", MTLGPUFamily.apple7), ("apple8", .apple8), ("apple9", .apple9)] {
  print("supports \(name): \(device.supportsFamily(family))")
}

let body = mode == "texture"
  ? """
    kernel void k(texture2d<ulong, access::read_write> t [[texture(0)]], uint2 id [[thread_position_in_grid]]) {
      t.atomic_max(id, ulong(id.x) << 40);
    }
    """
  : """
    kernel void k(device atomic_ulong *p [[buffer(0)]], uint id [[thread_position_in_grid]]) {
      atomic_max_explicit(p, ulong(id) << 40, memory_order_relaxed);
    }
    """
let source = "#include <metal_stdlib>\nusing namespace metal;\n" + body

let options = MTLCompileOptions()
options.languageVersion = .version3_1
do {
  let library = try device.makeLibrary(source: source, options: options)
  let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "k")!)
  if mode == "buffer" {
    let buffer = device.makeBuffer(length: 8, options: .storageModeShared)!
    memset(buffer.contents(), 0, 8)
    let queue = device.makeCommandQueue()!
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(buffer, offset: 0, index: 0)
    enc.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    let got = buffer.contents().load(as: UInt64.self)
    print("max: \(String(got, radix: 16)) expected: \(String(UInt64(255) << 40, radix: 16)) \(got == UInt64(255) << 40 ? "PASS" : "FAIL")")
  } else {
    print("pipeline built: PASS")
  }
} catch {
  print("FAIL: \(error)")
}
