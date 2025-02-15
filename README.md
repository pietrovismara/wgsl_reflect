Extracts metadata from WGSL shaders.

Since this is not a fully fledged WGSL parser, visibility for resources is specified via the optional custom `@resources` attribute on entry points.

Custom attributes can be stripped from the shader via the `comply` procedure, making the shader compliant to the spec.

```odin
shader := `
struct Camera {
	viewProjectionMatrix: mat4x4f,
}
@group(0) @binding(0)
var<uniform> camera: Camera;
@group(1) @binding(0)
var<storage> transform: array<mat4x4f>;

@group(1) @binding(1)
var tex: texture_2d<f32>;
@group(1) @binding(2)
var samp: sampler;

@vertex @resources(camera, transform)
fn v_main(@location(0) position: vec3f, @builtin(instance_index) iidx: u32) -> @builtin(position) vec4f {
	return camera.viewProjectionMatrix * transform[iidx] * vec4f(input.position, 1.0);
}

@fragment @resources(tex, samp)
fn f_main() -> @location(0) vec4f {
	let texColor = textureSample(tex, samp, vec2f(0, 0));
	return texColor;
}
`

main :: proc() {
	metadata, errors := wgsl.process(shader, context.temp_allocator)
	defer free_all(context.temp_allocator)

	assert(len(errors) == 0)

	{
		assert("camera" in metadata.resources)
		resource := metadata.resources["camera"]
		assert(resource.group == 0)
		assert(resource.binding == 0)
		assert(resource.type == .Buffer)
		assert(resource.buffer.type == .Uniform)
		assert(resource.visibility == {.Vertex})
	}

	{
		assert("transform" in metadata.resources)
		resource := metadata.resources["transform"]
		assert(resource.group == 1)
		assert(resource.binding == 0)
		assert(resource.type == .Buffer)
		assert(resource.buffer.type == .ReadOnlyStorage)
		assert(resource.visibility == {.Vertex})
	}

	{
		assert("tex" in metadata.resources)
		resource := metadata.resources["tex"]
		assert(resource.type == .Texture)
		assert(resource.texture.viewDimension == ._2D)
		assert(resource.texture.sampleType == .Float)
		assert(resource.texture.multisampled == false)
		assert(resource.visibility == {.Fragment})
	}

	{
		assert("samp" in metadata.resources)
		resource := metadata.resources["samp"]
		assert(resource.type == .Sampler)
		assert(resource.sampler.type == .Filtering)
		assert(resource.visibility == {.Fragment})
	}

	// Strips the shader of custom attributes like @stage
	spec_compliant_shader := wgsl.comply(shader)
}
```
