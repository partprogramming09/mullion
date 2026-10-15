/*
 Replays two EXP_MESH_DUMP files' triangles through a mesh pipeline shaped like Mullion's tessellation + GS stage,
 into a 64×192 two-layer RG32Float target with additive blending: B alone, then A, then B, each in its own waited
 command buffer, printing the pixels each B draw covers.
 Usage: metal_mesh_replay <dump A> <dump B> [parts=28] [workloads per part=8] [primitives per workload=128] [modes]
        [Mullion pixel shader .metallib]
 modes: "nolayer" (no per-primitive layer), "same" (B instead of A in between), "regs" (Mullion's vertex outputs:
 invariant position, reg1_0-3 and reg2_0-3), "mullionps" (with "regs": the given Mullion pixel shader)
 */
import Foundation
import Metal

let arguments = CommandLine.arguments
let parts = arguments.count > 3 ? Int(arguments[3])! : 28
let perPart = arguments.count > 4 ? Int(arguments[4])! : 8
let prims = arguments.count > 5 ? Int(arguments[5])! : 128
let modes = Set((arguments.count > 6 ? arguments[6] : "").split(separator: ",").map(String.init))
let (width, height) = (64, 192)
let stride = 40

// a dump's triangles, `stride` words each at (workload × prims + primitive): per vertex its output registers 0, 1 and
// 2 (12 words), then each vertex's layer (3 words) and a 1 where the slot has a triangle
func table(_ path: String) -> (words: [UInt32], workloads: Int) {
  let data = try! Data(contentsOf: URL(fileURLWithPath: path))
  let all = data.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
  let size = Int(all[1])
  let records = (all.count - 4) / size
  let workloads = (0..<records).map { Int(all[4 + $0 * size]) }.max()! + 1
  var words = [UInt32](repeating: 0, count: workloads * prims * stride)
  for i in 0..<records {
    let r = Array(all[(4 + i * size)..<(4 + (i + 1) * size)])
    let (workload, primitive, vertex) = (Int(r[0]), Int(r[1]), Int(r[2]))
    let at = (workload * prims + primitive) * stride
    // registers: 0 position, 1 location xyz and layer, 2 value
    words[at + vertex * 12 ..< at + vertex * 12 + 12] = r[4..<16]
    words[at + 36 + vertex] = r[11]
    words[at + 39] = 1
  }
  return (words, workloads)
}

let source = """
#include <metal_stdlib>
using namespace metal;
#if REGS
struct V {
  float4 position [[position, invariant]];
  float r10 [[user(reg1_0)]];
  float r11 [[user(reg1_1)]];
  float r12 [[user(reg1_2)]];
  uint r13 [[user(reg1_3)]];
  float r20 [[user(reg2_0)]];
  float r21 [[user(reg2_1)]];
  float r22 [[user(reg2_2)]];
  float r23 [[user(reg2_3)]];
};
#else
struct V {
  float4 position [[position]];
  float3 location [[flat]];
  float value;
};
#endif
struct PL { uint layer [[render_target_array_index]]; };
// Mullion's tessellation payload is 10128 bytes; the grid's first workload is its only word read here
struct Payload { uint words[10128 / 4]; };

[[object]] void object(object_data Payload &payload [[payload]], mesh_grid_properties grid,
                       uint lane [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]],
                       constant uint4 &shape [[buffer(0)]]) {
  if (lane == 0) {
    uint first = tg.z * shape.y;
    uint count = min(shape.y, max(shape.x, first) - first);
    payload.words[0] = first;
    grid.set_threadgroups_per_grid(uint3(count, shape.z, 1));
  }
}

using ML = mesh<V, PL, 3, 3, topology::triangle>;
using MN = mesh<V, void, 3, 3, topology::triangle>;
static void put_layer(thread ML &m, uint k, uint layer) { PL p; p.layer = layer; m.set_primitive(k, p); }
static void put_layer(thread MN &, uint, uint) {}
static float f(device const uint *w, uint i) { return as_type<float>(w[i]); }

// thread 0 makes the strip of the slot's triangle, writing each vertex's layer to primitive k as Mullion does
template <typename M>
static void emit(thread M &m, object_data const Payload &payload, uint3 tg, uint lane, constant uint4 &shape,
                 device const uint *table) {
  if (lane != 0) return;
  device const uint *slot = table + ((payload.words[0] + tg.x) * shape.z + tg.y) * 40;
  uint count = 0;
  if (slot[39]) {
    for (uint k = 0; k < 3; k++) {
      device const uint *w = slot + k * 12;
      V v;
      v.position = float4(f(w, 0), f(w, 1), f(w, 2), f(w, 3));
#if REGS
      v.r10 = f(w, 4); v.r11 = f(w, 5); v.r12 = f(w, 6); v.r13 = w[7];
      v.r20 = f(w, 8); v.r21 = f(w, 9); v.r22 = f(w, 10); v.r23 = f(w, 11);
#else
      v.location = float3(f(w, 4), f(w, 5), f(w, 6));
      v.value = f(w, 8);
#endif
      m.set_vertex(k, v);
      put_layer(m, k, slot[36 + k]);
      uint even = (k & 1) == 0;
      m.set_index(3 * k, k);
      m.set_index(3 * k + 1, k + 2 - even);
      m.set_index(3 * k + 2, k + 1 + even);
    }
    count = 1;
  }
  m.set_primitive_count(count);
}

[[mesh]] void layered(ML m, object_data const Payload &payload [[payload]], uint3 tg [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_threadgroup]], constant uint4 &shape [[buffer(0)]],
                      device const uint *table [[buffer(1)]]) {
  emit<ML>(m, payload, tg, lane, shape, table);
}
[[mesh]] void single(MN m, object_data const Payload &payload [[payload]], uint3 tg [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_threadgroup]], constant uint4 &shape [[buffer(0)]],
                     device const uint *table [[buffer(1)]]) {
  emit<MN>(m, payload, tg, lane, shape, table);
}
#if REGS
struct FI {
  float4 position [[position]];
  float r10 [[user(reg1_0), flat]];
  float r20 [[user(reg2_0)]];
};
struct FL { FI v; PL p; };
fragment float2 shade_layered(FL in [[stage_in]]) { return float2(1, 1 + in.v.r20 * 0); }
fragment float2 shade_single(FI in [[stage_in]]) { return float2(1, 1 + in.r20 * 0); }
#else
struct FL { V v; PL p; };
fragment float2 shade_layered(FL in [[stage_in]]) { return float2(1, 1 + in.v.value * 0); }
fragment float2 shade_single(V in [[stage_in]]) { return float2(1, 1 + in.value * 0); }
#endif
"""

let device = MTLCreateSystemDefaultDevice()!
let options = MTLCompileOptions()
options.preprocessorMacros = ["REGS": NSNumber(value: modes.contains("regs") ? 1 : 0)]
options.preserveInvariance = modes.contains("regs")
let library = try device.makeLibrary(source: source, options: options)
let layered = !modes.contains("nolayer")
let d = MTLMeshRenderPipelineDescriptor()
d.objectFunction = library.makeFunction(name: "object")
d.meshFunction = library.makeFunction(name: layered ? "layered" : "single")
if modes.contains("mullionps") {
  let path = URL(fileURLWithPath: arguments[7])
  let mullion = try device.makeLibrary(URL: path)
  d.fragmentFunction = mullion.makeFunction(name: mullion.functionNames[0])
} else {
  d.fragmentFunction = library.makeFunction(name: layered ? "shade_layered" : "shade_single")
}
d.colorAttachments[0].pixelFormat = .rg32Float
d.colorAttachments[0].isBlendingEnabled = true
d.colorAttachments[0].rgbBlendOperation = .add
d.colorAttachments[0].sourceRGBBlendFactor = .one
d.colorAttachments[0].destinationRGBBlendFactor = .one
d.payloadMemoryLength = 0
let (pipeline, _) = try device.makeRenderPipelineState(descriptor: d, options: [])

let target = MTLTextureDescriptor()
target.textureType = .type2DArray
target.pixelFormat = .rg32Float
target.width = width
target.height = height
target.arrayLength = 2
target.usage = [.renderTarget]
target.storageMode = .shared
let texture = device.makeTexture(descriptor: target)!
let queue = device.makeCommandQueue()!
// the Mullion pixel shader's two buffer arguments, which the layered one does not read
let unused = device.makeBuffer(length: 4096, options: .storageModeShared)!

// draws one dump's triangles in a fresh pass over a cleared target, waits, and returns the pixels covered in any layer
func draw(_ dump: (words: [UInt32], workloads: Int)) -> Int {
  let table = device.makeBuffer(bytes: dump.words, length: dump.words.count * 4, options: .storageModeShared)!
  let pass = MTLRenderPassDescriptor()
  pass.colorAttachments[0].texture = texture
  pass.colorAttachments[0].loadAction = .clear
  pass.colorAttachments[0].storeAction = .store
  pass.renderTargetArrayLength = 2
  let commands = queue.makeCommandBuffer()!
  let encoder = commands.makeRenderCommandEncoder(descriptor: pass)!
  encoder.setRenderPipelineState(pipeline)
  var shape: [UInt32] = [UInt32(dump.workloads), UInt32(perPart), UInt32(prims), 0]
  encoder.setObjectBytes(&shape, length: 16, index: 0)
  encoder.setMeshBytes(&shape, length: 16, index: 0)
  encoder.setMeshBuffer(table, offset: 0, index: 1)
  for index in 0..<2 { encoder.setFragmentBuffer(unused, offset: 0, index: index) }
  encoder.drawMeshThreadgroups(MTLSize(width: 1, height: 1, depth: parts),
                               threadsPerObjectThreadgroup: MTLSize(width: 8, height: 4, depth: 1),
                               threadsPerMeshThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
  encoder.endEncoding()
  commands.commit()
  commands.waitUntilCompleted()
  if let error = commands.error { print("error \(error)") }
  var covered = Set<Int>()
  var pixels = [Float](repeating: 0, count: width * height * 2)
  for layer in 0..<2 {
    texture.getBytes(&pixels, bytesPerRow: width * 8, bytesPerImage: width * height * 8,
                     from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, slice: layer)
    for i in 0..<(width * height) where pixels[2 * i] > 0 { covered.insert(i) }
  }
  return covered.count
}

let a = table(arguments[1]), b = table(arguments[2])
print("device \(device.name) | A \(a.workloads) workloads, B \(b.workloads) | parts \(parts) × \(perPart) workloads × \(prims) primitives | modes \(modes.sorted())")
let alone = draw(b)
_ = draw(modes.contains("same") ? b : a)
let after = draw(b)
print("B alone \(alone) pixels, B after \(modes.contains("same") ? "B" : "A") \(after) pixels: \(alone - after) missing")
