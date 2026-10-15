/*
 Draws d3d12_tessellator's layered patches with Mullion's own compiled functions (EXP_AIR_DUMP metallibs: object
 vshs_main, mesh ds_main, a pixel shader) in pure Metal, binding what Mullion binds: the root signature's buffer with
 the factors' constant buffer, the draw arguments and an empty index buffer view. Draws B alone, then A, then B, each
 in its own waited command buffer, into a 64×192 two-layer RG32Float target with additive blending.
 Usage: metal_mullion_replay <object .metallib> <mesh .metallib> <pixel .metallib> [A factors] [B factors] [modes]
 factors: 6 comma-separated values (edges, then inside); modes: "nocull", "same" (B instead of A in between),
 "private" (private target), "mullionenv" (Mullion's arena, fences, residency set and explicit states)
 */
import Foundation
import Metal

let arguments = CommandLine.arguments
let parse = { (s: String) -> [Float] in s.split(separator: ",").map { Float($0)! } }
let factorsA = arguments.count > 4 ? parse(arguments[4]) : [64, 64, 64, 64, 64, 33]
let factorsB = arguments.count > 5 ? parse(arguments[5]) : [2, 16.5, 2, 2, 12, 33]
let modes = Set((arguments.count > 6 ? arguments[6] : "").split(separator: ",").map(String.init))
let (width, height, parts) = (64, 192, 28)

let device = MTLCreateSystemDefaultDevice()!
// the one function of a Mullion metallib
func function(_ path: String) throws -> MTLFunction {
  let library = try device.makeLibrary(URL: URL(fileURLWithPath: path))
  return library.makeFunction(name: library.functionNames[0])!
}
let d = MTLMeshRenderPipelineDescriptor()
d.objectFunction = try function(arguments[1])
d.meshFunction = try function(arguments[2])
d.fragmentFunction = try function(arguments[3])
d.colorAttachments[0].pixelFormat = .rg32Float
d.colorAttachments[0].isBlendingEnabled = true
for c in [d.colorAttachments[0]!] {
  c.rgbBlendOperation = .add; c.alphaBlendOperation = .add
  c.sourceRGBBlendFactor = .one; c.destinationRGBBlendFactor = .one
  c.sourceAlphaBlendFactor = .one; c.destinationAlphaBlendFactor = .one
}
d.payloadMemoryLength = 0
d.objectThreadgroupSizeIsMultipleOfThreadExecutionWidth = true
d.meshThreadgroupSizeIsMultipleOfThreadExecutionWidth = true
let (pipeline, _) = try device.makeRenderPipelineState(descriptor: d, options: [])

let target = MTLTextureDescriptor()
target.textureType = .type2DArray
target.pixelFormat = .rg32Float
target.width = width
target.height = height
target.arrayLength = 2
target.usage = [.renderTarget, .shaderRead]
target.storageMode = modes.contains("private") ? .private : .shared
let texture = device.makeTexture(descriptor: target)!
let readback = device.makeBuffer(length: width * height * 8 * 2, options: .storageModeShared)!
let queue = device.makeCommandQueue()!
let constants = device.makeBuffer(length: 4096, options: .storageModeShared)!
let points = device.makeBuffer(length: 1 << 20, options: .storageModeShared)!
let locks = device.makeBuffer(length: 1 << 20, options: .storageModeShared)!
let sparse = device.makeBuffer(length: 4096, options: .storageModeShared)!
// the root signature: the constant buffer, the points UAV and the atomic locks, as GPU addresses
var root: [UInt64] = [constants.gpuAddress, points.gpuAddress, locks.gpuAddress]
let rootBuffer = device.makeBuffer(bytes: &root, length: 24, options: .storageModeShared)!
// a draw of 1 control point and 1 instance; the index buffer view of a draw without indices (R16_UINT)
var draw: [UInt32] = [1, 1, 0, 0, 0]
let drawBuffer = device.makeBuffer(bytes: &draw, length: 20, options: .storageModeShared)!
var view: [UInt32] = [0, 0, 0, 57]
let viewBuffer = device.makeBuffer(bytes: &view, length: 16, options: .storageModeShared)!

/* "mullionenv": Mullion's recording arena (1 GiB of host memory in one untracked buffer) holding the root signature,
   index buffer view, patch size and draw arguments at Mullion's offsets, a fence between passes, a residency set on
   the queue, and the depth, blend and viewport state Mullion sets */
let env = modes.contains("mullionenv")
let arenaSize = 1 << 30
let arena: MTLBuffer? = env ? {
  let memory = mmap(nil, arenaSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)!
  return device.makeBuffer(bytesNoCopy: memory, length: arenaSize,
                           options: [.storageModeShared, .hazardTrackingModeUntracked], deallocator: nil)
}() : nil
let (rootAt, viewAt, sizeAt, drawAt) = (0x10000, 0x10020, 0x10030, 0x10040)
if let arena {
  memcpy(arena.contents() + rootAt, &root, 24)
  memcpy(arena.contents() + viewAt, &view, 16)
  var patchSize: UInt32 = 1
  memcpy(arena.contents() + sizeAt, &patchSize, 4)
  memcpy(arena.contents() + drawAt, &draw, 20)
  let set = try device.makeResidencySet(descriptor: MTLResidencySetDescriptor())
  for buffer in [arena, constants, points, locks, readback] { set.addAllocation(buffer) }
  set.addAllocation(texture)
  set.commit()
  queue.addResidencySet(set)
}
let fence = device.makeFence()!
let depthState = device.makeDepthStencilState(descriptor: MTLDepthStencilDescriptor())!

// draws one patch in a fresh pass over a cleared target, waits, and returns the pixels covered in any layer
func drawPatch(_ factors: [Float]) -> Int {
  var values = factors + [Float](repeating: 0, count: 8 - factors.count)
  memcpy(constants.contents(), &values, 32)
  let pass = MTLRenderPassDescriptor()
  pass.colorAttachments[0].texture = texture
  pass.colorAttachments[0].loadAction = .clear
  pass.colorAttachments[0].storeAction = .store
  pass.renderTargetArrayLength = 2
  let commands = queue.makeCommandBuffer()!
  let encoder = commands.makeRenderCommandEncoder(descriptor: pass)!
  if env { encoder.waitForFence(fence, before: [.vertex, .object, .mesh]) }
  encoder.setRenderPipelineState(pipeline)
  if !modes.contains("nocull") {
    encoder.setCullMode(.back)
    encoder.setFrontFacing(.clockwise)
  }
  if let arena {
    encoder.setDepthStencilState(depthState)
    encoder.setDepthClipMode(.clip)
    for (index, offset) in [(0, rootAt), (1, 0)] {
      encoder.setObjectBuffer(arena, offset: offset, index: index)
      encoder.setMeshBuffer(arena, offset: offset, index: index)
      encoder.setFragmentBuffer(arena, offset: offset, index: index)
    }
    encoder.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(width), height: Double(height), znear: 0, zfar: 1))
    encoder.setScissorRect(MTLScissorRect(x: 0, y: 0, width: width, height: height))
    encoder.setBlendColor(red: 1, green: 1, blue: 1, alpha: 1)
    encoder.setStencilReferenceValue(0)
    for (index, offset) in [(3, viewAt), (8, sizeAt), (4, drawAt)] { encoder.setObjectBuffer(arena, offset: offset, index: index) }
  } else {
    for (index, buffer) in [(0, rootBuffer), (28, sparse)] {
      encoder.setObjectBuffer(buffer, offset: 0, index: index)
      encoder.setMeshBuffer(buffer, offset: 0, index: index)
      encoder.setFragmentBuffer(buffer, offset: 0, index: index)
    }
    encoder.setObjectBuffer(viewBuffer, offset: 0, index: 3)
    encoder.setObjectBuffer(drawBuffer, offset: 0, index: 4)
    encoder.useResources([constants, points, locks], usage: [.read, .write], stages: [.object, .mesh, .fragment])
  }
  encoder.drawMeshThreadgroups(MTLSize(width: 1, height: 1, depth: parts),
                               threadsPerObjectThreadgroup: MTLSize(width: 8, height: 4, depth: 1),
                               threadsPerMeshThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
  if env { encoder.updateFence(fence, after: .fragment) }
  encoder.endEncoding()
  // each layer copied to the readback buffer, as the test's CopyTextureRegion does
  let blit = commands.makeBlitCommandEncoder()!
  if env { blit.waitForFence(fence) }
  for layer in 0..<2 {
    blit.copy(from: texture, sourceSlice: layer, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
              sourceSize: MTLSize(width: width, height: height, depth: 1), to: readback,
              destinationOffset: layer * width * height * 8, destinationBytesPerRow: width * 8,
              destinationBytesPerImage: width * height * 8)
  }
  if env { blit.updateFence(fence) }
  blit.endEncoding()
  commands.commit()
  commands.waitUntilCompleted()
  if let error = commands.error { print("error \(error)") }
  var covered = Set<Int>()
  let pixels = readback.contents().bindMemory(to: Float.self, capacity: width * height * 4)
  for layer in 0..<2 {
    for i in 0..<(width * height) where pixels[layer * width * height * 2 + 2 * i] > 0 { covered.insert(i) }
  }
  return covered.count
}

print("device \(device.name) | A \(factorsA) B \(factorsB) | modes \(modes.sorted())")
let alone = drawPatch(factorsB)
let middle = drawPatch(modes.contains("same") ? factorsB : factorsA)
let after = drawPatch(factorsB)
print("B alone \(alone) pixels, in between \(middle), B after \(after): \(alone - after) missing")
