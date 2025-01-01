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

        @group(0) @binding(0) var<uniform> uniforms : Uniforms;
        @group(0) @binding(1) var<storage, read_write> path_state_1 : PathState_0;
        @group(0) @binding(2) var<storage, read_write> path_state_2 : PathState_1;

        @group(1) @binding(0) var<storage, read_write> queues : QueuesStage2;

        ${params.scene.kernels.getHitInfoCode(2)}

        const Pi      = 3.14159265358979323846;
        const InvPi   = 0.31830988618379067154;
        const Inv2Pi  = 0.15915494309189533577;
        const Inv4Pi  = 0.07957747154594766788;
        const PiOver2 = 1.57079632679489661923;
        const PiOver4 = 0.78539816339744830961;
        const Sqrt2   = 1.41421356237309504880;

        var<workgroup> wg_stage_3_queue_size : array<atomic<i32>, 2>;
        var<workgroup> wg_nearest_hit_queue : array<i32, ${WG_SIZE}>;
        var<workgroup> wg_any_hit_queue : array<i32, ${WG_SIZE}>;

        ${getLambertDiffuseBRDF()}
        ${getEmissiveBRDF()}
        ${getPerfectMirrorBRDF()}

        
        fn area_light_sample_li(
            light_o : vec3f,
            light_le : vec3f,
            light_for : vec3f,
            light_right : vec3f,
            light_scale : vec2f,
            wi : ptr<function, vec3f>,
            dist : ptr<function, f32>,
            o : vec3f,
            r2 : vec2f,
        ) -> vec4f {
            var area : f32 = 4.f * (light_scale.x * light_scale.y);

            // generate the sample point
            var sample_point_local : vec2f = (r2 - vec2f(.5f)) * 2.f * light_scale;
            var sample_point : vec3f = light_o + light_right * sample_point_local.x + normalize(cross(light_for, light_right)) * sample_point_local.y;

            var del_pos : vec3f = sample_point - o;
            var light_dist : f32 = length(del_pos);

            // calculate outgoing direction and pdf
            *wi = del_pos / light_dist;
            var local_pdf : f32 = light_dist * light_dist / (area * abs(dot(light_for, *wi)));

            *dist = light_dist;

            return vec4f(light_le, local_pdf);
        }

        fn sample_f(
            wi : ptr<function, vec3f>,
            random_seed : ptr<function, f32>,
            flags : ptr<function, u32>,
            wo : vec3f,
            material_index : i32,
        ) -> vec4f {
            switch(material_index) {
                case 0: {
                    return lambert_diffuse_sample_f(wo, wi, random_seed, vec3f(.9f), flags);
                }
                case 1: {
                    return emissive_sample_f(wo, wi, random_seed, 25.f * vec3f(1., 1., 0.9), flags);
                }
                case 2: {
                    //return lambert_diffuse_sample_f(wo, wi, random_seed, vec3f(.5f, 0.f, 0.f), flags);
                    return perfect_mirror_sample_f(wo, wi, vec3f(.8), flags);
                }
                case 3: {
                    return lambert_diffuse_sample_f(wo, wi, random_seed, vec3f(0.f, .5f, 0.f), flags);
                }
                default: {
                    return vec4f(0.f);
                }
            }
        }

        fn f(
            wo : vec3f,
            wi : vec3f,
            material_index : i32
        ) -> vec3f {
            switch(material_index) {
                case 0: {
                    return lambert_diffuse_f(wo, wi, vec3f(.9f));
                }
                case 1: {
                    return emissive_f(wo, wi, 25.f * vec3f(1., 1., 0.9));
                }
                case 2: {
                    //return lambert_diffuse_f(wo, wi, vec3f(.5f, 0.f, 0.f));
                    return perfect_mirror_f(wo, wi, vec3f(1.f));
                }
                case 3: {
                    return lambert_diffuse_f(wo, wi, vec3f(0.f, .5f, 0.f));
                }
                default: {
                    return vec3f(0.f);
                }
            }
        }

        @compute @workgroup_size(${WG_SIZE})
        fn main(@builtin(global_invocation_id) global_id : vec3u, @builtin(local_invocation_id) local_id : vec3u) {
            var queue_idx : i32 = i32(global_id.x);
            if (queue_idx >= queues.stage_2_queue_size[1]) {
                
            } else {
                // load in all parameters from path state
                var path_idx : i32 = queues.material_queue[queue_idx];

                var flags : u32 = 1u;

                var o : vec3f = path_state_1.path_o[path_idx];
                var d : vec3f = path_state_1.path_d[path_idx];

                var hit_obj : i32 = path_state_1.hit_obj[path_idx];
                var hit_tri : i32 = path_state_1.hit_tri[path_idx];

                var random_seed : f32 = path_state_1.random_seed[path_idx];
                var material_index : i32 = objects[hit_obj].material;

                // get hit information and calculate local frame
                var hit_info : TriangleHitInfo = get_triangle_hit_info(o, d, hit_obj, hit_tri);

                var hit_pos = o + d * hit_info.dist;
                var hit_nor : vec3f = hit_info.normal;

                if (dot(hit_nor, d) > 0.f) {
                    hit_nor = -hit_nor;
                }

                var o1 : vec3f = normalize(ortho(hit_nor));
                var o2 : vec3f = normalize(cross(o1, hit_nor));

                var wo : vec3f = to_local(o1, o2, hit_nor, -d);

                // sample next direction and brdf
                var wi : vec3f;
                var brdf_pdf : vec4f = sample_f(&wi, &random_seed, &flags, wo, material_index);

                d = to_world(o1, o2, hit_nor, wi);

                // if this was not a delta material, evaluate direct lighting
                if ((flags & (1u << 3u)) == 0u) {
                    var light_dist : f32;
                    var light_dir : vec3f;

                    var le_pdf : vec4f = area_light_sample_li(
                        vec3f(0.f, 0.f, 9.999),
                        vec3f(25.f),
                        vec3f(0., 0., -1.),
                        vec3f(0., 1., 0.),
                        vec2f(2., 2.),
                        &light_dir,
                        &light_dist,
                        hit_pos,
                        rand2(random_seed)
                    ); random_seed += 2.f;

                    var local_light_dir : vec3f = to_local(o1, o2, hit_nor, light_dir);

                    var f : vec3f = f(wo, local_light_dir, material_index) * local_light_dir.z;

                    var ld : vec3f = f * le_pdf.xyz / le_pdf.w;

                    if (any(ld > vec3f(0.f))) {
                        var l_idx : i32 = atomicAdd(&wg_stage_3_queue_size[1], 1);
                        wg_any_hit_queue[l_idx] = path_idx;

                        path_state_2.nee_direction_distance[path_idx] = vec4f(light_dir, light_dist);
                        path_state_2.nee_ld[path_idx] = vec4f(ld, 3.1415);
                    }
                }

                path_state_1.material_throughput_pdf[path_idx] = brdf_pdf;
                path_state_1.flags[path_idx] |= flags;
                path_state_1.random_seed[path_idx] = random_seed;
                path_state_1.path_o[path_idx] = hit_pos + hit_nor * 1e-5;
                path_state_1.path_d[path_idx] = d;

                // only continue this path if it has a next valid direction
                if (length(d) > 0.f) {
                    var l_idx : i32 = atomicAdd(&wg_stage_3_queue_size[0], 1);
                    wg_nearest_hit_queue[l_idx] = path_idx;
                }
            }

            workgroupBarrier();

            // if this is the first thread in the work group, copy local queues to global memory
            if (local_id.x == 0u) {
                {
                    var num_writes = atomicLoad(&wg_stage_3_queue_size[0]);
                    if (num_writes > 0) {
                        var offset : i32 = atomicAdd(&queues.stage_3_queue_size[0], num_writes);
                        for (var x = 0; x < num_writes; x++) {
                            queues.nearest_hit_queue[offset + x] = wg_nearest_hit_queue[x];
                        }
                    }
                }
                {
                    var num_writes = atomicLoad(&wg_stage_3_queue_size[1]);
                    if (num_writes > 0) {
                        var offset : i32 = atomicAdd(&queues.stage_3_queue_size[1], num_writes);
                        for (var x = 0; x < num_writes; x++) {
                            queues.any_hit_queue[offset + x] = wg_any_hit_queue[x];
                        }
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