function initLogicKernel(params) {
    const device = params.device

    const WG_SIZE = 64

    const SM = device.createShaderModule({
        code: SRC(),
        label: "logic shader module"
    })

    const PIPELINE = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [
                params.bindGroupLayouts.pathState, 
                params.bindGroupLayouts.queues,
                params.bindGroupLayouts.image
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
        P.setBindGroup(2, params.bindGroups.image)
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

        @group(1) @binding(0) var<storage, read_write> queues : QueuesStage1;

        @group(2) @binding(0) var<storage, read_write> image : array<atomic<i32>>;

        var<workgroup> wg_stage_2_queue_size : array<atomic<i32>, 2>;

        var<workgroup> wg_camera_queue : array<i32, ${WG_SIZE}>;
        var<workgroup> wg_material_queue : array<i32, ${WG_SIZE}>;

        @compute @workgroup_size(${WG_SIZE})
        fn main(@builtin(global_invocation_id) global_id : vec3u, @builtin(local_invocation_id) local_id : vec3u) {
            var path_idx : i32 = i32(global_id.x);

            // check if this path needs to get a new pixel index
            var b_get_new_pixel_index : bool = false;
            if (path_idx >= ${params.numPaths}) {
                
            } else {
                if (uniforms.first_sample > 0) {
                    // special case if this is the first execution of this kernel
                    path_state.random_seed[path_idx] = f32(baseHash(vec2u(u32(path_idx), u32(path_idx + 1)))) / f32(0xffffffffu) + .008;
                    path_state.path_throughput[path_idx] = vec3f(1.f);

                    var l_idx : i32 = atomicAdd(&wg_stage_2_queue_size[0], 1);
                    wg_camera_queue[l_idx] = path_idx;

                    b_get_new_pixel_index = true;
                } else {
                    // otherwise, see if the path needs to be restarted or continued
                    var path_throughput : vec3f = path_state.path_throughput[path_idx];
                    var num_bounces : i32 = path_state.num_bounces[path_idx] + 1;

                    var path_contribution : vec4f = vec4f(0.f);

                    var b_hit : bool = path_state.hit_obj[path_idx] >= 0;
                    var b_new_path : bool = false;

                    var flags : u32 = path_state.flags[path_idx];


                    if ((flags & 1u) != 0u) {
                        // then a material hit occurred
                        path_throughput *= path_state.material_throughput_pdf[path_idx].xyz;

                        if (num_bounces > 3) {
                            var r2 : vec2f = rand2(path_state.random_seed[path_idx]);
                            path_state.random_seed[path_idx] += 2.f;

                            var q : f32 = min(max(.1, 1. - path_throughput.y), .7);
                            if (r2.x < q) {
                                path_throughput = vec3f(0.f);
                            } else {
                                path_throughput = path_throughput / (1.f - q);
                            }
                        }
                    }

                    if (num_bounces > 20) {
                        path_throughput = vec3f(0.f);
                    }

                    if (!b_hit) {
                        path_contribution += vec4f(2.f * path_throughput, 0.f);
                        path_throughput = vec3f(0.f);
                    }

                    // then this path was terminated
                    if (all(path_throughput == vec3f(0.f)))  {
                        path_contribution.w = 1.f;

                        var l_idx : i32 = atomicAdd(&wg_stage_2_queue_size[0], 1);
                        wg_camera_queue[l_idx] = path_idx;

                        num_bounces = 0;
                        path_throughput = vec3f(1.f);

                        b_get_new_pixel_index = true;
                    } else {
                        var l_idx : i32 = atomicAdd(&wg_stage_2_queue_size[1], 1);
                        wg_material_queue[l_idx] = path_idx;

                        num_bounces += 1;
                    }

                    path_state.path_throughput[path_idx] = path_throughput;
                    path_state.num_bounces[path_idx] = num_bounces;
                    path_state.flags[path_idx] = 0u;

                    // add light accumulation to image
                    if (any(path_contribution != vec4f(0.f))) {
                        f32_atomic_add(&image[4 * path_state.pixel_index[path_idx] + 0], path_contribution.x);
                        f32_atomic_add(&image[4 * path_state.pixel_index[path_idx] + 1], path_contribution.y);
                        f32_atomic_add(&image[4 * path_state.pixel_index[path_idx] + 2], path_contribution.z);
                        f32_atomic_add(&image[4 * path_state.pixel_index[path_idx] + 3], path_contribution.w);                        
                    }
                }
            }

            // check
            if (b_get_new_pixel_index) {
                path_state.pixel_index[path_idx] = path_idx;
            }

            workgroupBarrier();

            // if this is the first thread in the work group, copy local queues to local memory
            if (local_id.x == 0u) {

                var num_camera_writes   : i32 = atomicLoad(&wg_stage_2_queue_size[0]);
                var num_material_writes : i32 = atomicLoad(&wg_stage_2_queue_size[1]);

                if (num_camera_writes > 0) {
                    var offset : i32 = atomicAdd(&queues.stage_2_queue_size[0], num_camera_writes);

                    for (var x = 0; x < num_camera_writes; x++) {
                        queues.camera_queue[offset + x] = wg_camera_queue[x];
                    }
                }

                if (num_material_writes > 0) {
                    var offset : i32 = atomicAdd(&queues.stage_2_queue_size[1], num_material_writes);

                    for (var x = 0; x < num_material_writes; x++) {
                        queues.material_queue[offset + x] = wg_material_queue[x];
                    }
                }
            }
        }

        fn f32_atomic_add(source : ptr<storage, atomic<i32>, read_write>, add : f32) {
            var pre_val : i32;
            var new_val : i32;

            pre_val = atomicLoad(source);
            new_val = bitcast<i32>(bitcast<f32>(pre_val) + add);

            while (!atomicCompareExchangeWeak(source, pre_val, new_val).exchanged) {
                pre_val = atomicLoad(source);
                new_val = bitcast<i32>(bitcast<f32>(pre_val) + add);
            }
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