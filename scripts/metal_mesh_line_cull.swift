// Draws a 5-vertex line strip from a mesh shader with lines 1 and 2 culled, and prints which lines reach the target.
import Metal

let source = """
#include <metal_stdlib>
using namespace metal;
struct V { float4 position [[position]]; };
struct P { uint value [[flat]]; bool culled [[primitive_culled]]; };
using M = mesh<V, P, 8, 8, topology::line>;
struct F { V v; P p; };

static float4 at(uint x) { return float4((x + 0.5) / 8 * 2 - 1, 1 - 3.5 / 8 * 2, 0, 1); }
static bool behind(uint v) { return v >= 1 && v <= 3; }

// each primitive's culled flag written once, culling the primitives `mask` has a bit for
static void strip(M m, uint mask) {
  for (uint v = 0; v < 5; v++) { V o; o.position = at(v); m.set_vertex(v, o); }
  for (uint p = 0; p < 4; p++) {
    m.set_index(2 * p, p);
    m.set_index(2 * p + 1, p + 1);
    P q; q.value = p + 1; q.culled = (mask >> p) & 1;
    m.set_primitive(p, q);
  }
  m.set_primitive_count(4);
}
[[mesh]] void none(M m) { strip(m, 0); }
[[mesh]] void only1(M m) { strip(m, 2); }
[[mesh]] void clean(M m) { strip(m, 6); }

// the order of Mullion's line strip: per vertex, the indices of primitive k and the culled flag of primitive k - 1
[[mesh]] void mullion(M m, uint tid [[thread_index_in_threadgroup]]) {
  for (uint k = 0; k < 5; k++) {
    V o; o.position = at(k); m.set_vertex(k, o);
    m.set_index(2 * k, k);
    m.set_index(2 * k + 1, k + 1);
    uint p = k >= 1 ? k - 1 : k;
    P q; q.value = p + 1; q.culled = k >= 1 && behind(k) && behind(k - 1);
    m.set_primitive(p, q);
  }
  m.set_primitive_count(4);
}

// lines 2 and 3 dropped by giving both their ends the same vertex instead of a culled flag
[[mesh]] void degenerate(M m) {
  for (uint v = 0; v < 5; v++) { V o; o.position = at(v); m.set_vertex(v, o); }
  for (uint p = 0; p < 4; p++) {
    bool dropped = behind(p) && behind(p + 1);
    m.set_index(2 * p, p);
    m.set_index(2 * p + 1, dropped ? p : p + 1);
    P q; q.value = p + 1; q.culled = false;
    m.set_primitive(p, q);
  }
  m.set_primitive_count(4);
}

// the same in Mullion's order: vertex k rewrites the second index of primitive k - 1 once it knows it is dropped
[[mesh]] void mullion_degenerate(M m) {
  for (uint k = 0; k < 5; k++) {
    V o; o.position = at(k); m.set_vertex(k, o);
    m.set_index(2 * k, k);
    m.set_index(2 * k + 1, k + 1);
    if (k >= 1) {
      bool dropped = behind(k) && behind(k - 1);
      m.set_index(2 * (k - 1) + 1, dropped ? k - 1 : k);
      P q; q.value = k; q.culled = false;
      m.set_primitive(k - 1, q);
    }
  }
  m.set_primitive_count(4);
}

fragment uint shade(F in [[stage_in]]) { return in.p.value; }
"""

let device = MTLCreateSystemDefaultDevice()!
let library = try device.makeLibrary(source: source, options: nil)
let target = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Uint, width: 8, height: 8, mipmapped: false)
target.usage = [.renderTarget, .shaderRead]
target.storageMode = .shared
print("device:", device.name)

for name in ["none", "only1", "clean", "mullion", "degenerate", "mullion_degenerate"] {
  let descriptor = MTLMeshRenderPipelineDescriptor()
  descriptor.meshFunction = library.makeFunction(name: name)
  descriptor.fragmentFunction = library.makeFunction(name: "shade")
  descriptor.colorAttachments[0].pixelFormat = .r32Uint
  let (pipeline, _) = try device.makeRenderPipelineState(descriptor: descriptor, options: [])
  let texture = device.makeTexture(descriptor: target)!
  let pass = MTLRenderPassDescriptor()
  pass.colorAttachments[0].texture = texture
  pass.colorAttachments[0].loadAction = .clear
  pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
  pass.colorAttachments[0].storeAction = .store
  let commands = device.makeCommandQueue()!.makeCommandBuffer()!
  let encoder = commands.makeRenderCommandEncoder(descriptor: pass)!
  encoder.setRenderPipelineState(pipeline)
  encoder.drawMeshThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerObjectThreadgroup: MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerMeshThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
  encoder.endEncoding()
  commands.commit()
  commands.waitUntilCompleted()
  var row = [UInt32](repeating: 0, count: 8)
  texture.getBytes(&row, bytesPerRow: 32, from: MTLRegionMake2D(0, 3, 8, 1), mipmapLevel: 0)
  let lines = Set(row.filter { $0 != 0 }).sorted()
  print("\(name): row 3 = \(row), lines drawn = \(lines)")
}
