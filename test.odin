package wgsl_reflect

import "core:fmt"
import "core:testing"

@(private = "file")
shader := `
struct Camera {
	viewProjectionMatrix: mat4x4f,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> time: f32; // Should be ignored
@group(1) @binding(0) var<storage> transform: array<mat4x4f>;
@group(1) @binding(1) var tex: texture_2d<f32>;
@group(1) @binding(2) var samp: sampler;

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


@(private = "file")
shader_compliant: cstring = `
struct Camera {
	viewProjectionMatrix: mat4x4f,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> time: f32; // Should be ignored
@group(1) @binding(0) var<storage> transform: array<mat4x4f>;
@group(1) @binding(1) var tex: texture_2d<f32>;
@group(1) @binding(2) var samp: sampler;

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


@(test)
test :: proc(t: ^testing.T) {
	metadata, errs := process(shader, context.temp_allocator)
	defer free_all(context.temp_allocator)

	testing.expect(t, len(errs) == 0)

	{
		testing.expect(t, "camera" in metadata.resources)
		camera := metadata.resources["camera"]
		testing.expect(t, camera.group == 0)
		testing.expect(t, camera.binding == 0)
		testing.expect(t, camera.type == .Buffer)
		testing.expect(t, camera.buffer.type == .Uniform)
		testing.expect(t, camera.visibility == {.Vertex})
	}

	{
		testing.expect(t, "transform" in metadata.resources)
		transform := metadata.resources["transform"]
		testing.expect(t, transform.group == 1)
		testing.expect(t, transform.binding == 0)
		testing.expect(t, transform.type == .Buffer)
		testing.expect(t, transform.buffer.type == .ReadOnlyStorage)
		testing.expect(t, transform.visibility == {.Vertex})
	}

	{
		testing.expect(t, "tex" in metadata.resources)
		resource := metadata.resources["tex"]
		testing.expect(t, resource.type == .Texture)
		testing.expect(t, resource.texture.viewDimension == ._2D)
		testing.expect(t, resource.texture.sampleType == .Float)
		testing.expect(t, resource.texture.multisampled == false)
		testing.expect(t, resource.visibility == {.Fragment})
	}

	{
		testing.expect(t, "samp" in metadata.resources)
		resource := metadata.resources["samp"]
		testing.expect(t, resource.type == .Sampler)
		testing.expect(t, resource.sampler.type == .Filtering)
		testing.expect(t, resource.visibility == {.Fragment})
	}

	testing.expect(t, shader_compliant == comply(shader, metadata, context.temp_allocator))
}
