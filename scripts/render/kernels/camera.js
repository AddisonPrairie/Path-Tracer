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
                params.bindGroupLayouts.queues
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
        P.setBindGroup(1, params.bindGroups.queues)
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

        @group(1) @binding(0) var<storage, read_write> queues : QueuesStage2;

        const Pi      = 3.14159265358979323846;
        const InvPi   = 0.31830988618379067154;
        const Inv2Pi  = 0.15915494309189533577;
        const Inv4Pi  = 0.07957747154594766788;
        const PiOver2 = 1.57079632679489661923;
        const PiOver4 = 0.78539816339744830961;
        const Sqrt2   = 1.41421356237309504880;

        var<workgroup> wg_stage_3_queue_size : atomic<i32>;
        var<workgroup> wg_ray_trace_queue : array<i32, ${WG_SIZE}>;

        @compute @workgroup_size(${WG_SIZE})
        fn main(@builtin(global_invocation_id) global_id : vec3u, @builtin(local_invocation_id) local_id : vec3u) {
            var queue_idx : i32 = i32(global_id.x);
            if (queue_idx >= queues.stage_2_queue_size[0]) {
                
            } else {
                // compute camera ray
                var path_idx : i32 = queues.camera_queue[queue_idx];
                var pixel_idx : i32 = path_state.pixel_index[path_idx];
                var coord : vec2f = vec2f(vec2i(pixel_idx % uniforms.image_size.x, pixel_idx / uniforms.image_size.x)) + rand2(path_state.random_seed[path_idx]);
                path_state.random_seed[path_idx] += 2.f;

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
                var num_writes = atomicLoad(&wg_stage_3_queue_size);

                if (num_writes > 0) {
                    var offset : i32 = atomicAdd(&queues.stage_3_queue_size[0], num_writes);
                    for (var x = 0; x < num_writes; x++) {
                        queues.ray_trace_queue[offset + x] = wg_ray_trace_queue[x];
                    }
                }
            }
        }
        
        fn getCameraRay(coord : vec2f, o : ptr<function, vec3f>, d : ptr<function, vec3f>) {
            var sspace : vec2f = coord / vec2f(uniforms.image_size);
            sspace = sspace * 2.f - vec2f(1.f);
            sspace.y *= -1.f;

            var camera_aspect_ratio : f32 = f32(uniforms.image_size.x) / f32(uniforms.image_size.y);

            var local : vec3f = vec3f(
                camera_aspect_ratio * sspace.x * sin(uniforms.camera_fov * Pi / 180.f),
                1.f,
                sspace.y * sin(uniforms.camera_fov * Pi / 180.f)
            );

            var forward : vec3f = normalize(uniforms.camera_look_at - uniforms.camera_position);
            var   right : vec3f = normalize(vec3f(forward.y, -forward.x, 0.));
            var      up : vec3f = cross(right, forward);

            *o = uniforms.camera_position;
            *d = toWorld(right, forward, up, normalize(local));
        }

        fn toWorld(v_x : vec3f, v_y : vec3f, v_z : vec3f, w : vec3f) -> vec3f {
            return v_x * w.x + v_y * w.y + v_z * w.z;
        }
        
        fn baseHash(p : vec2u) -> u32 {
            var p2 : vec2u = 1103515245u*((p >> vec2u(1u))^(p.yx));
            var h32 : u32 = 1103515245u*((p2.x)^(p2.y>>3u));
            return h32^(h32 >> 16u);
        }
        fn rand2(seed : f32) -> vec2f {
            var n : u32 = baseHash(bitcast<vec2u>(vec2f(seed + 1., seed + 2.)));
            var rz : vec2u = vec2u(n, n * 48271u);
            return vec2f(rz.xy & vec2u(0x7fffffffu))/f32(0x7fffffff);
        }`
    }
}