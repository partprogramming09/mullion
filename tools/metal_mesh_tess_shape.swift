// Counts the pixels a SIDE×SIDE target gets through a mesh pipeline shaped like Mullion's tessellation + GS stage.
// Args: [grid height=256] [active rows=128] [modes: invariant,packed,shared,nozero,prior=H:A,...] [side=64]
// [extra float4 varyings=0]
import Metal

let arguments = CommandLine.arguments
let perWorkload = arguments.count > 1 ? Int(arguments[1])! : 256
let active = arguments.count > 2 ? Int(arguments[2])! : 128
let modes = Set((arguments.count > 3 ? arguments[3] : "").split(separator: ",").map(String.init))
let side = arguments.count > 4 ? Int(arguments[4])! : 64
let extra = arguments.count > 5 ? Int(arguments[5])! : 0
let draws = Int(modes.first(where: { $0.hasPrefix("draws=") })?.dropFirst(6) ?? "1") ?? 1
let pixels = side * side
let triangles = modes.contains("shared") ? pixels / 2 : pixels
let workloads = (triangles + active - 1) / active
let perGroup = max(1, 1024 / perWorkload)
let groups = (workloads + perGroup - 1) / perGroup

let source = """
#include <metal_stdlib>
using namespace metal;
struct V {
#if INVARIANT
  float4 position [[position, invariant]];
#else
  float4 position [[position]];
#endif
  float3 location [[flat]];
  float value;
#if PACKED
  uint packed [[flat]];
#endif
#if SCALARS
  float s0;
  float s1;
#endif
#if EXTRA > 0
  float4 e0;
#endif
#if EXTRA > 1
  float4 e1;
#endif
#if EXTRA > 2
  float4 e2;
#endif
#if EXTRA > 3
  float4 e3;
#endif
};
struct PL { uint layer [[render_target_array_index]]; };
struct Payload { uint words[PAYLOAD_WORDS]; };

static float4 corner(uint cell, uint i) {
#if SHARED
  float2 base = float2(cell / 2 % (SIDE / 2), cell / 2 / (SIDE / 2)) * 2;
  float2 offset = cell % 2 == 0 ? (i == 0 ? float2(0, 0) : i == 1 ? float2(2, 0) : float2(0, 2))
                                : (i == 0 ? float2(2, 0) : i == 1 ? float2(2, 2) : float2(0, 2));
  float2 at = (base + offset) / (SIDE / 2.0) - 1.0;
#else
  float2 center = float2(cell % SIDE, cell / SIDE) + 0.5;
  float2 offset = i == 0 ? float2(-0.4, -0.4) : i == 1 ? float2(0.6, -0.4) : float2(-0.4, 0.6);
  float2 at = (center + offset) / (SIDE / 2.0) - 1.0;
#endif
  return float4(at.x, -at.y, 0, 1);
}
static uint layer_of(uint cell) { return ((cell * 2654435761u) >> 31) & 1; }

[[object]] void object(object_data Payload &payload [[payload]], mesh_grid_properties grid,
                       uint lane [[thread_index_in_threadgroup]], uint3 group3 [[threadgroup_position_in_grid]],
                       constant uint4 &shape [[buffer(0)]]) {
  // the object grid runs along x, or along z as Mullion's tessellation parts do
  uint group = group3.x + group3.z + shape.w;
  if (!NOZERO)
    for (uint i = lane; i < PAYLOAD_WORDS; i += 32) payload.words[i] = 0;
  if (lane == 0) {
#if EMPTY
    // every other object threadgroup asks for an empty grid, as Mullion's parts with no workloads left do
    payload.words[0] = group / 2;
    uint3 size = group % 2 ? uint3(0, shape.y, 1) : uint3(shape.x, shape.y, 1);
#else
    payload.words[0] = group;
    uint3 size = uint3(shape.x, shape.y, 1);
#endif
#if GRID3D
    // the rows split over y and z, as Mullion puts geometry shader instances on z
    size = uint3(size.x, size.y / 2, 2);
#endif
    grid.set_threadgroups_per_grid(size);
  }
}

using ML = mesh<V, PL, 3, 3, topology::triangle>;
using MN = mesh<V, void, 3, 3, topology::triangle>;
static void put_layer(thread ML &m, uint k, uint cell) { PL p; p.layer = layer_of(cell); m.set_primitive(k, p); }
static void put_layer(thread MN &, uint, uint) {}

template <typename M>
static void emit(thread M &m, object_data const Payload &payload, uint3 tg, uint lane, constant uint4 &shape) {
  if (lane != 0) return;
#if GRID3D
  tg.y = tg.y + tg.z * (shape.y / 2);
#endif
  uint cell = (payload.words[0] * shape.x + tg.x) * shape.z + tg.y;
  uint count = 0;
  if (tg.y < shape.z && cell < TRIANGLES) {
    for (uint k = 0; k < 3; k++) {
      V v; v.position = corner(cell, k); v.location = float3(cell, 0, 0); v.value = 1;
#if SCALARS
      v.s0 = k;
      v.s1 = cell;
#endif
#if PACKED
      v.packed = layer_of(cell);
#endif
#if EXTRA > 0
      v.e0 = float4(k);
#endif
#if EXTRA > 1
      v.e1 = float4(k);
#endif
#if EXTRA > 2
      v.e2 = float4(k);
#endif
#if EXTRA > 3
      v.e3 = float4(k);
#endif
      m.set_vertex(k, v);
      put_layer(m, k, cell);
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
                      uint lane [[thread_index_in_threadgroup]], constant uint4 &shape [[buffer(0)]]) {
  emit<ML>(m, payload, tg, lane, shape);
}
[[mesh]] void single(MN m, object_data const Payload &payload [[payload]], uint3 tg [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_threadgroup]], constant uint4 &shape [[buffer(0)]]) {
  emit<MN>(m, payload, tg, lane, shape);
}
// every varying is read and the result depends on it, so none is dropped from the mesh's memory block
static uint used(V v) {
#if SCALARS
  // only the two scalars, so the block holds what Mullion's layered mesh holds
  float s = v.s0 + v.s1;
#else
  float s = v.location.x + v.value;
#endif
#if EXTRA > 0
  s += v.e0.x + v.e0.y + v.e0.z + v.e0.w;
#endif
#if EXTRA > 1
  s += v.e1.x + v.e1.y + v.e1.z + v.e1.w;
#endif
#if EXTRA > 2
  s += v.e2.x + v.e2.y + v.e2.z + v.e2.w;
#endif
#if EXTRA > 3
  s += v.e3.x + v.e3.y + v.e3.z + v.e3.w;
#endif
  return s > -1e30 ? 1 : 2;
}
struct FL { V v; PL p; };
fragment uint shade_layered(FL in [[stage_in]]) { return used(in.v); }
fragment uint shade_single(V in [[stage_in]]) { return used(in); }
"""

let device = MTLCreateSystemDefaultDevice()!
let options = MTLCompileOptions()
options.preprocessorMacros = [
  "TRIANGLES": NSNumber(value: triangles),
  "SIDE": NSNumber(value: side),
  "EXTRA": NSNumber(value: extra),
  "INVARIANT": NSNumber(value: modes.contains("invariant") ? 1 : 0),
  "PACKED": NSNumber(value: modes.contains("packed") ? 1 : 0),
  "SHARED": NSNumber(value: modes.contains("shared") ? 1 : 0),
  "EMPTY": NSNumber(value: modes.contains("empty") ? 1 : 0),
  "SCALARS": NSNumber(value: modes.contains("scalar2") ? 1 : 0),
  "GRID3D": NSNumber(value: modes.contains("grid3d") ? 1 : 0),
  "NOZERO": NSNumber(value: modes.contains("nozero") ? 1 : 0),
  // Mullion's tessellation payload is 10128 bytes, with the pipeline's payload length left at 0
  "PAYLOAD_WORDS": NSNumber(value: modes.contains("payload") ? 10128 / 4 : 2048),
]
options.preserveInvariance = modes.contains("invariant")
let library = try device.makeLibrary(source: source, options: options)
print("device \(device.name) | modes \(modes.sorted()) side \(side) extra \(extra) | \(groups) object groups × \(perGroup) workloads × \(perWorkload) rows (\(groups * perGroup * perWorkload) mesh groups)")

for (mesh, fragment) in [("single", "shade_single"), ("layered", "shade_layered")] {
  let target = MTLTextureDescriptor()
  target.textureType = .type2DArray
  target.pixelFormat = .r32Uint
  target.width = side
  target.height = side
  target.arrayLength = 2
  target.usage = [.renderTarget, .shaderRead]
  target.storageMode = .shared
  let d = MTLMeshRenderPipelineDescriptor()
  d.objectFunction = library.makeFunction(name: "object")
  d.meshFunction = library.makeFunction(name: mesh)
  d.fragmentFunction = library.makeFunction(name: fragment)
  d.colorAttachments[0].pixelFormat = .r32Uint
  d.payloadMemoryLength = modes.contains("payload") ? 0 : 8192
  d.objectThreadgroupSizeIsMultipleOfThreadExecutionWidth = true
  d.meshThreadgroupSizeIsMultipleOfThreadExecutionWidth = true
  let pipeline: MTLRenderPipelineState
  do {
    (pipeline, _) = try device.makeRenderPipelineState(descriptor: d, options: [])
  } catch {
    print("\(mesh): pipeline error \(error)")
    continue
  }
  let texture = device.makeTexture(descriptor: target)!
  let pass = MTLRenderPassDescriptor()
  pass.colorAttachments[0].texture = texture
  pass.colorAttachments[0].loadAction = .clear
  pass.colorAttachments[0].storeAction = .store
  pass.renderTargetArrayLength = 2
  let queue = device.makeCommandQueue()!
  // "prior=<grid height>:<active rows>" first draws that other shape with the same pipeline in its own command buffer,
  // committed and waited for, as a different patch drawn before the one Mullion's test checks
  if let prior = modes.first(where: { $0.hasPrefix("prior=") }) {
    let values = prior.dropFirst(6).split(separator: ":").map { Int($0)! }
    let (rows, rowsActive) = (values[0], values[1])
    let priorPerGroup = max(1, 1024 / rows)
    let priorGroups = ((triangles + rowsActive - 1) / rowsActive + priorPerGroup - 1) / priorPerGroup
    let priorCommands = queue.makeCommandBuffer()!
    let priorEncoder = priorCommands.makeRenderCommandEncoder(descriptor: pass)!
    priorEncoder.setRenderPipelineState(pipeline)
    var shape: [UInt32] = [UInt32(priorPerGroup), UInt32(rows), UInt32(rowsActive), 0]
    priorEncoder.setObjectBytes(&shape, length: 16, index: 0)
    priorEncoder.setMeshBytes(&shape, length: 16, index: 0)
    priorEncoder.drawMeshThreadgroups(MTLSize(width: priorGroups, height: 1, depth: 1),
                                      threadsPerObjectThreadgroup: MTLSize(width: 32, height: 1, depth: 1),
                                      threadsPerMeshThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    priorEncoder.endEncoding()
    priorCommands.commit()
    priorCommands.waitUntilCompleted()
  }
  var commands = queue.makeCommandBuffer()!
  var encoder = commands.makeRenderCommandEncoder(descriptor: pass)!
  encoder.setRenderPipelineState(pipeline)
  // the object groups split over `draws` consecutive draws, every other one indirect with "indirect"; with "cmdbufs"
  // each draw has its own command buffer, committed and waited for, as Mullion submits each test case
  let objectGroups = modes.contains("empty") ? groups * 2 : groups
  let perDraw = (objectGroups + draws - 1) / draws
  let objectThreads = modes.contains("obj2d") ? MTLSize(width: 1, height: 32, depth: 1) : MTLSize(width: 32, height: 1, depth: 1)
  let meshThreads = MTLSize(width: 32, height: 1, depth: 1)
  for first in stride(from: 0, to: objectGroups, by: perDraw) {
    if modes.contains("cmdbufs") && first > 0 {
      encoder.endEncoding()
      commands.commit()
      commands.waitUntilCompleted()
      pass.colorAttachments[0].loadAction = .load
      commands = queue.makeCommandBuffer()!
      encoder = commands.makeRenderCommandEncoder(descriptor: pass)!
      encoder.setRenderPipelineState(pipeline)
    }
    let count = min(perDraw, objectGroups - first)
    var shape: [UInt32] = [UInt32(perGroup), UInt32(perWorkload), UInt32(active), UInt32(first)]
    encoder.setObjectBytes(&shape, length: 16, index: 0)
    encoder.setMeshBytes(&shape, length: 16, index: 0)
    let grid = modes.contains("objz") ? MTLSize(width: 1, height: 1, depth: count) : MTLSize(width: count, height: 1, depth: 1)
    if modes.contains("indirect") && (first / perDraw) % 2 == 1 {
      var arguments: [UInt32] = [UInt32(grid.width), UInt32(grid.height), UInt32(grid.depth)]
      let buffer = device.makeBuffer(bytes: &arguments, length: 12, options: .storageModeShared)!
      encoder.drawMeshThreadgroups(indirectBuffer: buffer, indirectBufferOffset: 0,
                                   threadsPerObjectThreadgroup: objectThreads, threadsPerMeshThreadgroup: meshThreads)
    } else {
      encoder.drawMeshThreadgroups(grid, threadsPerObjectThreadgroup: objectThreads, threadsPerMeshThreadgroup: meshThreads)
    }
  }
  encoder.endEncoding()
  commands.commit()
  commands.waitUntilCompleted()
  if let error = commands.error { print("\(mesh): error \(error)"); continue }
  var layers = [[UInt32]](repeating: [UInt32](repeating: 0, count: pixels), count: 2)
  for layer in 0..<2 {
    texture.getBytes(&layers[layer], bytesPerRow: side * 4, bytesPerImage: pixels * 4,
                     from: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0, slice: layer)
  }
  var covered = 0, twice = 0
  for i in 0..<pixels {
    let n = Int(layers[0][i] > 0 ? 1 : 0) + Int(layers[1][i] > 0 ? 1 : 0)
    if n >= 1 { covered += 1 }
    if n >= 2 { twice += 1 }
  }
  print("  \(mesh): \(covered) of \(pixels) covered (\(pixels - covered) missing), \(twice) twice")
}
