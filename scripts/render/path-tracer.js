function initPathTracer(params) {

    console.log(params)


    function PATH_STATE_DESC() {
        return /* wgsl */ `
        struct PathState {
            pixel_index : array<i32, ${params.numActivePaths}>,
            num_bounces : array<i32, ${params.numActivePaths}>,

            path_throughput : array<vec3f, ${params.numActivePaths}>,

            material_throughput_pdf : array<vec4f, ${params.numActivePaths}>,

            path_o : array<vec3f, ${params.numActivePaths}>,
            path_d : array<vec3f, ${params.numActivePaths}>,

            hit_obj : array<i32, ${params.numActivePaths}>,
            hit_tri : array<i32, ${params.numActivePaths}>
        };`
    }
}