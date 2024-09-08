function initRayTraceKernel(params) {

    const WG_SIZE = 64


    function SRC() {
        return /* wgsl */ `
        @group(0) @binding(0) var<storage, read_write> path_state : PathState;

        @group(1) @binding(0) var<storage, read_write> ray_trace_queue_size : i32;
        @group(1) @binding(1) var<storage, read_write> ray_trace_queue : array<i32>;

        @compute @workgroup_size(${WG_SIZE})
        fn main(@builtin(global_invocation_id) global_id : vec3u) {
            var queue_idx : i32 = i32(global_id.x);
            if (queue_idx >= ray_trace_queue_size) {
                return;
            }

            var path_idx : i32 = ray_trace_queue[queue_idx];

            var o : vec3f = path_state.path_o[path_idx];
            var d : vec3f = path_state.path_d[path_idx];

            var res : BVHHitResult = intersect_bvh(o, d);

            path_state.hit_obj[path_idx] = res.hit_obj;
            path_state.hit_tri[path_idx] = res.hit_tri;
        }`
    }
}