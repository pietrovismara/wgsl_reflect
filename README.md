Extracts metadata from WGSL shaders.

Due to my unwillingness to implement a fully fledged WGSL parser, visibility for resources is specified via the optional custom `@stage` attribute.

Custom attributes can be stripped from the shader via the `strip` procedure, making the shader compliant to the spec.

```odin
shader := `
struct Camera {
	viewProjectionMatrix: mat4x4f,
}
@group(0) @binding(0) @stage(vertex)
var<uniform> camera: Camera;
@group(1) @binding(0) @stage(vertex)
var<storage> transform: array<mat4x4f>;

@group(1) @binding(1) @stage(fragment)
var tex: texture_2d<f32>;
@group(1) @binding(2) @stage(fragment)
var samp: sampler;

@vertex
fn v_main(@location(0) position: vec3f, @builtin(instance_index) iidx: u32) -> @builtin(position) vec4f {
	return camera.viewProjectionMatrix * transform[iidx] * vec4f(input.position, 1.0);
}

@fragment
fn f_main() -> @location(0) vec4f {
	let texColor = textureSample(tex, samp, vec2f(0, 0));
	return texColor;
}
`

main :: proc() {
	metadata, statements, tokens, errors := wgsl.process(shader, context.temp_allocator)
	defer free_all(context.temp_allocator)

	assert(len(errors) == 0)

	vertex_entry_point, ok := wgsl.get_entry_point(metadata, .Vertex)
	assert(ok && vertex_entry_point.name == "v_main")
	fragment_entry_point, ok := wgsl.get_entry_point(metadata, .Fragment)
	assert(ok && fragment_entry_point.name == "f_main")

	{
		assert("camera" in metadata.resources)
		resource := metadata.resources["camera"]
		assert(resource.group == 0)
		assert(resource.binding == 0)
		assert(resource.type == .Buffer)
		assert(resource.buffer.binding_type == .Uniform)
		assert(resource.visibility == {.Vertex})
	}

	{
		assert("transform" in metadata.resources)
		resource := metadata.resources["transform"]
		assert(resource.group == 1)
		assert(resource.binding == 0)
		assert(resource.type == .Buffer)
		assert(resource.buffer.binding_type == .ReadOnlyStorage)
		assert(resource.visibility == {.Vertex})
	}

	{
		assert("tex" in metadata.resources)
		resource := metadata.resources["tex"]
		assert(resource.type == .Texture)
		assert(resource.texture.dimension == ._2D)
		assert(resource.texture.sample_type == .Float)
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
	spec_compliant_shader := wgsl.strip(shader)
}
```
