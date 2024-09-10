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
                params.bindGroupLayouts.stage2Queues,
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
        P.setBindGroup(1, params.bindGroups.stage2Queues)
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

        @group(1) @binding(0) var<storage, read_write> stage_2_queue_size : array<i32>;
        @group(1) @binding(1) var<storage, read_write> camera_queue : array<i32>;
        @group(1) @binding(2) var<storage, read_write> material_queue : array<i32>;
        @group(1) @binding(3) var<storage, read_write> stage_3_queue_size : atomic<i32>;
        @group(1) @binding(4) var<storage, read_write> ray_trace_queue : i32;

        ${params.scene.kernels.getHitInfoCode(2)}

        var<workgroup> wg_ray_trace_queue : array<i32, ${WG_SIZE}>;

        @compute @workgroup_size(${WG_SIZE})
        fn main(@builtin(global_invocation_id) global_id : vec3u, @builtin(local_invocation_id) local_id : vec3u) {
            var queue_idx : i32 = i32(global_id.x);
            if (queue_idx >= stage_2_queue_size[0]) {
                
            } else {
                // compute camera ray
                var path_idx : i32 = camera_queue[queue_idx];

                var o : vec3f = path_state.path_o[path_idx];
                var d : vec3f = path_state.path_d[path_idx];

                var hit_obj : i32 = path_state.hit_obj[path_idx];
                var hit_tri : i32 = path_state.hit_tri[path_idx];

                var hit_info : TriangleHitInfo = get_triangle_hit_info(o, d, hit_obj, hit_tri);

                //path_state.path_o[path_idx] = o;
                //path_state.path_d[path_idx] = d;

                //wg_ray_trace_queue[local_id.x] = path_idx;
            }

            workgroupBarrier();

            // if this is the first thread in the work group, copy local ray trace queue to global memory
            if (local_id.x == 0u) {
                var offset : i32 = atomicAdd(&stage_3_queue_size, ${WG_SIZE});

                for (var x = 0; x < ${WG_SIZE}; x++) {
                    ray_trace_queue[offset + x] = wg_ray_trace_queue[x];
                }
            }
        }
        
        fn toLocal(v_x : vec3f, v_y : vec3f, v_z : vec3f, w : vec3f) -> vec3f {
            return vec3f(dot(v_x, w), dot(v_y, w), dot(v_z, w));
        }

        fn toWorld(v_x : vec3f, v_y : vec3f, v_z : vec3f, w : vec3f) -> vec3f {
            return v_x * w.x + v_y * w.y + v_z * w.z;
        }`
    }
}