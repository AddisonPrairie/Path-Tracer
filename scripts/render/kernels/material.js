function initMaterialKernel(params) {
    const device = params.device

    const WG_SIZE = 64

    const SM = device.createShaderModule({
        code: SRC(),
        label: "material shader module"
    })

    const PIPELINE = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [
                params.bindGroupLayouts.pathState, 
                params.bindGroupLayouts.queues,
                params.bindGroupLayouts.scene
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
        P.setBindGroup(2, params.bindGroups.scene)
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

        ${params.scene.kernels.getHitInfoCode(2)}

        const Pi      = 3.14159265358979323846;
        const InvPi   = 0.31830988618379067154;
        const Inv2Pi  = 0.15915494309189533577;
        const Inv4Pi  = 0.07957747154594766788;
        const PiOver2 = 1.57079632679489661923;
        const PiOver4 = 0.78539816339744830961;
        const Sqrt2   = 1.41421356237309504880;

        var<workgroup> wg_stage_3_queue_size : atomic<i32>;
        var<workgroup> wg_ray_trace_queue : array<i32, ${WG_SIZE}>;

        fn lambert_diffuse_sample_f(
            wo : vec3f, 
            wi : ptr<function, vec3f>, 
            seed : ptr<function, f32>, 
            albedo : vec3f,
            flags : ptr<function, u32>
        ) -> vec4f {
            *wi = cosineSampleHemisphere(rand2(*seed)); *seed += 2.f;
            return vec4f(pow(albedo, vec3f(2.2)), (*wi).z);
        }

        @compute @workgroup_size(${WG_SIZE})
        fn main(@builtin(global_invocation_id) global_id : vec3u, @builtin(local_invocation_id) local_id : vec3u) {
            var queue_idx : i32 = i32(global_id.x);
            if (queue_idx >= queues.stage_2_queue_size[1]) {
                
            } else {
                var path_idx : i32 = queues.material_queue[queue_idx];

                var flags : u32 = 1u;

                var o : vec3f = path_state.path_o[path_idx];
                var d : vec3f = path_state.path_d[path_idx];

                var hit_obj : i32 = path_state.hit_obj[path_idx];
                var hit_tri : i32 = path_state.hit_tri[path_idx];

                var hit_info : TriangleHitInfo = get_triangle_hit_info(o, d, hit_obj, hit_tri);

                var hit_pos = o + d * hit_info.dist;

                var o1 : vec3f = normalize(ortho(hit_info.normal));
                var o2 : vec3f = normalize(cross(o1, hit_info.normal));

                var wo : vec3f = to_local(o1, o2, hit_info.normal, -d);
                var random_seed : f32 = path_state.random_seed[path_idx];

                var brdf_pdf : vec4f;
                var wi : vec3f;

                var material_index : i32 = objects[hit_obj].material;

                if (material_index == 0) {
                    brdf_pdf = lambert_diffuse_sample_f(wo, &wi, &random_seed, vec3f(.5f), &flags);
                }
                if (material_index == 1) {
                    brdf_pdf = lambert_diffuse_sample_f(wo, &wi, &random_seed, vec3f(.2f), &flags);
                }

                d = to_world(o1, o2, hit_info.normal, wi);

                path_state.material_throughput_pdf[path_idx] = brdf_pdf;
                path_state.flags[path_idx] |= flags;
                path_state.random_seed[path_idx] = random_seed;
                path_state.path_o[path_idx] = hit_pos + hit_info.normal * .0001;
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

        // sampling functions
        fn cosineSampleHemisphere(r2 : vec2f) -> vec3f {
            var d : vec2f = uniformSampleDisk(r2);
            var z : f32 = sqrt(max(0., 1. - d.x * d.x - d.y * d.y));
            return vec3f(d.xy, z);
        }

        fn uniformSampleDisk(r2 : vec2f) -> vec2f {
            var r : f32 = sqrt(max(r2.x, 0.));
            var theta : f32 = 2. * Pi * r2.y;
            return vec2f(r * cos(theta), r * sin(theta));
        }

        // noise functions
        fn baseHash(p : vec2u) -> u32 {
            var p2 : vec2u = 1103515245u*((p >> vec2u(1u))^(p.yx));
            var h32 : u32 = 1103515245u*((p2.x)^(p2.y>>3u));
            return h32^(h32 >> 16u);
        }
        fn rand2(seed : f32) -> vec2f {
            var n : u32 = baseHash(bitcast<vec2u>(vec2f(seed + 1., seed + 2.)));
            var rz : vec2u = vec2u(n, n * 48271u);
            return vec2f(rz.xy & vec2u(0x7fffffffu))/f32(0x7fffffff);
        }

        // misc utility functions
        fn ortho(v : vec3<f32>) -> vec3<f32> {
            if (abs(v.x) > abs(v.y)) {
                return vec3<f32>(-v.y, v.x, 0.);
            }
            return  vec3<f32>(0., -v.z, v.y);
        }
        
        fn to_local(v_x : vec3f, v_y : vec3f, v_z : vec3f, w : vec3f) -> vec3f {
            return vec3f(dot(v_x, w), dot(v_y, w), dot(v_z, w));
        }

        fn to_world(v_x : vec3f, v_y : vec3f, v_z : vec3f, w : vec3f) -> vec3f {
            return v_x * w.x + v_y * w.y + v_z * w.z;
        }`
    }
}