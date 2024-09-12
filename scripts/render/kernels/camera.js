function initCameraKernel(params) {
    const device = params.device

    const WG_SIZE = 64

    const SM = device.createShaderModule({
        code: SRC(),
        label: "camera shader module"
    })

    const PIPELINE = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [
                params.bindGroupLayouts.pathState, 
                params.bindGroupLayouts.stage2Queues
            ]
        }),
        compute: {
            module: SM,
            entryPoint: "main"
        }
    })

    return { execute }

    async function execute() {
        const CE = device.createCommandEncoder()
        const  P = CE.beginComputePass()

        P.setPipeline(PIPELINE)
        P.setBindGroup(0, params.bindGroups.pathState)
        P.setBindGroup(1, params.bindGroups.stage2Queues)
        P.dispatchWorkgroups(Math.ceil(params.numPaths / WG_SIZE))
        P.end()

        device.queue.submit([CE.finish()])
        await device.queue.onSubmittedWorkDone()
    }

    function SRC() {
        return /* wgsl */ `
        ${params.sharedStructCode}

        @group(0) @binding(0) var<storage, read_write> path_state : PathState;
        @group(0) @binding(1) var<uniform> uniforms : Uniforms;

        @group(1) @binding(0) var<storage, read_write> stage_2_queue_size : array<i32>;
        @group(1) @binding(1) var<storage, read_write> camera_queue : array<i32>;
        @group(1) @binding(2) var<storage, read_write> material_queue : array<i32>;
        @group(1) @binding(3) var<storage, read_write> stage_3_queue_size : atomic<i32>;
        @group(1) @binding(4) var<storage, read_write> ray_trace_queue : i32;

        var<workgroup> wg_stage_3_queue_size : atomic<i32>;
        var<workgroup> wg_ray_trace_queue : array<i32, ${WG_SIZE}>;

        @compute @workgroup_size(${WG_SIZE})
        fn main(@builtin(global_invocation_id) global_id : vec3u, @builtin(local_invocation_id) local_id : vec3u) {
            var queue_idx : i32 = i32(global_id.x);
            if (queue_idx >= stage_2_queue_size[0]) {
                
            } else {
                // compute camera ray
                var path_idx : i32 = camera_queue[queue_idx];
                var pixel_idx : i32 = path_state.pixel_index[path_idx];
                var coord : vec2f = vec2f(vec2i(pixel_idx % uniforms.image_size.x, pixel_idx / uniforms.image_size.x));

                var o : vec3f;
                var d : vec3f;

                getCameraRay(coord, &o, &d);

                path_state.path_o[path_idx] = o;
                path_state.path_d[path_idx] = d;

                var l_idx : i32 = atomicAdd(&wg_stage_3_queue_size, 1);
                wg_ray_trace_queue[l_idx] = path_idx;
            }

            workgroupBarrier();

            // if this is the first thread in the work group, copy local ray trace queue to global memory
            if (local_id.x == 0u) {
                var offset : i32 = atomicAdd(&stage_3_queue_size, ${WG_SIZE});

                var num_writes = atomicLoad(&wg_stage_3_queue_size);
                for (var x = 0; x < num_writes; x++) {
                    ray_trace_queue[offset + x] = wg_ray_trace_queue[x];
                }
            }
        }
        
        fn getCameraRay(coord : vec2f, o : ptr<function, vec3f>, d : ptr<function, vec3f>) {
            var sspace : vec2f = coord / vec2f(uniforms.image_size);
            sspace = sspace * 2.f - vec2f(1.f);
            sspace.y *= -1.f;

            var camera_aspect_ratio : f32 = f32(uniforms.image_size.x) / f32(uniforms.image_size.y);

            var local : vec3f = vec3f(
                camera_aspect_ratio * sspace.x * sin(uniforms.camera_fov),
                1.f,
                sspace.y * sin(uniforms.camera_fov)
            );

            var forward : vec3f = normalize(uniforms.camera_look_at - uniforms.camera_position);
            var   right : vec3f = normalize(vec3f(forward.y, -forward.x, 0.));
            var      up : vec3f = cross(right, forward);

            *o = uniforms.camera_position;
            *d = toWorld(right, forward, up, normalize(local));
        }

        fn toWorld(v_x : vec3f, v_y : vec3f, v_z : vec3f, w : vec3f) -> vec3f {
            return v_x * w.x + v_y * w.y + v_z * w.z;
        }`
    }
}