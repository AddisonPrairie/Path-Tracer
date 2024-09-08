function initPathTracer(device, scene) {


    function PATH_STATE_DESC(numActivePaths) {
        return /* wgsl */ `
        struct PathState {
            pixel_index : array<i32, ${numActivePaths}>,
            num_bounces : array<i32, ${numActivePaths}>,

            path_throughput : array<vec3f, ${numActivePaths}>,

            material_throughput_pdf : array<vec4f, ${numActivePaths}>,

            path_o : array<vec3f, ${numActivePaths}>,
            path_d : array<vec3f, ${numActivePaths}>,

            hit_obj : array<i32, ${numActivePaths}>,
            hit_tri : array<i32, ${numActivePaths}>
        };`
    }
}