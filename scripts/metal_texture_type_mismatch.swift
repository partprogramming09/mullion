// Reads a 2D texture through a texture2d_array handle (and a 1-layer 2D array view through a texture2d one), as a
// bindless D3D12 descriptor read by a shader that declares the other type; prints each (slice, lod) read.
import Metal

let device = MTLCreateSystemDefaultDevice()!
print("device: \(device.name)")
let source = """
#include <metal_stdlib>
using namespace metal;
struct Handles { texture2d_array<float> as_array; texture2d<float> as_2d; texture2d_array<float> matching; };
kernel void k(device const Handles &h [[buffer(0)]], device float *out [[buffer(1)]], uint j [[thread_position_in_grid]]) {
  constexpr sampler s(filter::nearest, mip_filter::nearest);
  uint slice = j % 4, lod = j / 4;
  out[j] = h.as_array.read(uint2(0, 0), slice, lod).r * 255.0;
  out[16 + j] = h.as_2d.read(uint2(0, 0), lod).r * 255.0;
  out[32 + j] = h.as_array.sample(s, float2(0.5), slice, level(lod)).r * 255.0;
  out[48 + j] = h.as_2d.sample(s, float2(0.5), level(lod)).r * 255.0;
  out[64 + j] = h.as_array.gather(s, float2(0.5), slice).x * 255.0;
  out[80 + j] = h.matching.read(uint2(0, 0), slice, lod).r * 255.0;
  out[96 + j] = h.matching.sample(s, float2(0.5), slice, level(lod)).r * 255.0;
}
"""
let library = try device.makeLibrary(source: source, options: nil)
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "k")!)

/// A 256×128 R8Unorm texture of `type` with 8 mips and `layers` layers; texel value = layer * 8 + mip + 1.
func texture(_ type: MTLTextureType, layers: Int) -> MTLTexture {
  let d = MTLTextureDescriptor()
  d.textureType = type
  d.pixelFormat = .r8Unorm
  d.width = 256
  d.height = 128
  d.mipmapLevelCount = 8
  d.arrayLength = layers
  d.usage = .shaderRead
  let t = device.makeTexture(descriptor: d)!
  for layer in 0..<layers {
    for mip in 0..<8 {
      let w = max(256 >> mip, 1), h = max(128 >> mip, 1)
      let bytes = [UInt8](repeating: UInt8(layer * 8 + mip + 1), count: w * h)
      t.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: mip, slice: layer, withBytes: bytes,
                bytesPerRow: w, bytesPerImage: w * h)
    }
  }
  return t
}

let plain = texture(.type2D, layers: 1)
let layered = texture(.type2DArray, layers: 2)
let oneLayer = layered.makeTextureView(pixelFormat: .r8Unorm, textureType: .type2DArray,
                                       levels: 0..<8, slices: 1..<2)!
// the fix's shape: the 2D texture seen through a 1-layer 2D array view, as a texture2d_array
let matching = plain.makeTextureView(pixelFormat: .r8Unorm, textureType: .type2DArray, levels: 0..<8, slices: 0..<1)!
let handles = device.makeBuffer(length: 24, options: .storageModeShared)!
handles.contents().storeBytes(of: plain.gpuResourceID, as: MTLResourceID.self)
handles.contents().storeBytes(of: oneLayer.gpuResourceID, toByteOffset: 8, as: MTLResourceID.self)
handles.contents().storeBytes(of: matching.gpuResourceID, toByteOffset: 16, as: MTLResourceID.self)
let out = device.makeBuffer(length: 112 * 4, options: .storageModeShared)!

let queue = device.makeCommandQueue()!
let cb = queue.makeCommandBuffer()!
let enc = cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(pipeline)
enc.setBuffer(handles, offset: 0, index: 0)
enc.setBuffer(out, offset: 0, index: 1)
enc.useResources([plain, oneLayer, matching], usage: .read)
enc.dispatchThreads(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1))
enc.endEncoding()
cb.commit()
cb.waitUntilCompleted()

let v = out.contents().bindMemory(to: Float.self, capacity: 112)
print("2D texture read as texture2d_array (D3D wants mip + 1 for slice 0, else 0):")
for j in 0..<16 {
  let slice = j % 4, lod = j / 4
  let want = slice == 0 ? Float(lod + 1) : 0
  print(String(format: "  slice %d lod %d: got %3.0f want %3.0f %@", slice, lod, v[j], want, v[j] == want ? "" : "WRONG"))
}
print("1-layer 2D array view (layer 1) read as texture2d (D3D wants 8 + lod + 1):")
for lod in 0..<4 {
  let want = Float(8 + lod + 1)
  print(String(format: "  lod %d: got %3.0f want %3.0f %@", lod, v[16 + lod * 4], want, v[16 + lod * 4] == want ? "" : "WRONG"))
}
// sampling clamps the slice to the view's layers (D3D11.3 7.18.9), so every slice of the 1-layer 2D gives mip + 1
print("2D texture sampled as texture2d_array at level(lod) (D3D wants lod + 1, slice clamped):")
for j in 0..<16 {
  let want = Float(j / 4 + 1)
  print(String(format: "  slice %d lod %d: got %3.0f want %3.0f %@", j % 4, j / 4, v[32 + j], want, v[32 + j] == want ? "" : "WRONG"))
}
print("1-layer 2D array view sampled as texture2d at level(lod) (D3D wants 8 + lod + 1):")
for lod in 0..<4 {
  let want = Float(8 + lod + 1)
  print(String(format: "  lod %d: got %3.0f want %3.0f %@", lod, v[48 + lod * 4], want, v[48 + lod * 4] == want ? "" : "WRONG"))
}
print("2D texture gathered as texture2d_array (mip 0; D3D wants 1, slice clamped):")
for slice in 0..<4 {
  print(String(format: "  slice %d: got %3.0f want   1 %@", slice, v[64 + slice], v[64 + slice] == 1 ? "" : "WRONG"))
}
var wrong = 0
print("control: 1-layer 2D array view of the 2D texture, read and sampled as texture2d_array:")
for j in 0..<16 {
  let slice = j % 4, lod = j / 4
  let wantRead: Float = slice == 0 ? Float(lod + 1) : 0, wantSample = Float(lod + 1)
  wrong += (v[80 + j] == wantRead ? 0 : 1) + (v[96 + j] == wantSample ? 0 : 1)
  print(String(format: "  slice %d lod %d: read %3.0f (want %3.0f), sample %3.0f (want %3.0f)",
               slice, lod, v[80 + j], wantRead, v[96 + j], wantSample))
}
print("control: \(wrong == 0 ? "PASS" : "FAIL, \(wrong) wrong values")")
