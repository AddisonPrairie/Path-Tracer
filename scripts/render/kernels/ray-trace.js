function initRayTraceKernel(params) {
    const device = params.device

    const WG_SIZE = 64

    const SM = device.createShaderModule({
        code: SRC(),
        label: "ray trace shader module"
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

        @group(1) @binding(0) var<storage, read_write> queues : QueuesStage3;

        ${params.scene.kernels.getNearestHitCode(2)}

        @compute @workgroup_size(${WG_SIZE})
        fn main(@builtin(global_invocation_id) global_id : vec3u) {
            var queue_idx : i32 = i32(global_id.x);
            if (queue_idx >= queues.stage_3_queue_size[0]) {
                return;
            }

            var path_idx : i32 = queues.ray_trace_queue[queue_idx];

            var o : vec3f = path_state.path_o[path_idx];
            var d : vec3f = path_state.path_d[path_idx];

            var res : BVHHitResult = intersect_bvh(o, d);

            path_state.hit_obj[path_idx] = res.hit_obj;
            path_state.hit_tri[path_idx] = res.hit_tri;
        }`
    }
}