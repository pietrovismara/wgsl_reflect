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

@vertex
fn main(@location(0) position: vec3f, @builtin(instance_index) iidx: u32) -> vec4f {
	return camera.viewProjectionMatrix * transform[iidx] * vec4f(input.position, 1.0);
}
`

main :: proc() {
	metadata, statements, tokens, errors := wgsl.process(shader, context.temp_allocator)
	defer free_all(context.temp_allocator)

	assert(len(errors) == 0)

	vertex_entry_point, ok := wgsl.get_entry_point(metadata, .Vertex)
	assert(ok && vertex_entry_point.name == "main")

	assert("transform" in metadata.resources)
	transform := metadata.resources["transform"]
	assert(transform.group == 1)
	assert(transform.binding == 0)
	assert(transform.type == .Buffer)
	assert(transform.space == .Storage)
	assert(transform.access == .Read)
	assert(transform.visibility == {.Vertex})

	// Strips the shader of custom attributes like @stage
	spec_compliant_shader := wgsl.strip(shader)
}
```
