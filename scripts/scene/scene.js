function initScene(device) {

    const builders = initBuilders(device)

    // externally editable state
    let meshes  = []
    let objects = []

    // internal state
    let sceneGPUState = null
    let sceneBindGroupInfo = null

    return { registerMesh, instanceMesh, build, kernels: { getNearestHitCode, getHitInfoCode, getSceneBindGroupInfo } }

    async function build() {
        // build BVHs for all meshes included in the scene
        let utilizedMeshOrder = {}
        for (var i = 0; i < objects.length; i++) {
            if (objects[i].type === "mesh") utilizedMeshOrder[objects[i].meshID] = {}
        }
        let utilizedMeshCount = 0
        let runningTriOffset  = 0
        for (var x in utilizedMeshOrder) {
            utilizedMeshOrder[x] = { order: utilizedMeshCount++, triangleOffset: runningTriOffset }
            if (!meshes[x].bvh) {
                meshes[x].bvh = await builders.buildMeshBVH(meshes[x].mesh)
            }
            runningTriOffset += meshes[x].bvh.numTriangles
        }

        // pack all triangle meshes into a single buffer, including rewriting pointers in BVHs
        let packedMeshes = await builders.packMeshes(meshes, utilizedMeshOrder)

        // compute transform matrices for all meshes
        for (var i = 0; i < objects.length; i++) {
            if (objects[i].type === "mesh") {
                objects[i].transformMatrices = computeTransformMatrices(objects[i].transform)
            }
        }
        
        // compute the bounding boxes of every object in the scene, and the overall scene bounds
        let bounds = { min: [1e30, 1e30, 1e30], max: [-1e30, -1e30, -1e30] }
        for (var i = 0; i < objects.length; i++) {
            if (objects[i].type === "mesh") {
                objects[i].bounds = getTransformedBoundingBox(objects[i].transformMatrices, meshes[objects[i].meshID].bvh.bounds)
            }
            if (objects[i].type === "sphere") {
                objects[i].bounds = {
                    min: [objects[i].position[0] - objects[i].radius, objects[i].position[1] - objects[i].radius, objects[i].position[2] - objects[i].radius],
                    max: [objects[i].position[0] + objects[i].radius, objects[i].position[1] + objects[i].radius, objects[i].position[2] + objects[i].radius],
                }
            }

            bounds.min[0] = Math.min(bounds.min[0], objects[i].bounds.min[0])
            bounds.min[1] = Math.min(bounds.min[1], objects[i].bounds.min[1])
            bounds.min[2] = Math.min(bounds.min[2], objects[i].bounds.min[2])

            bounds.max[0] = Math.max(bounds.max[0], objects[i].bounds.max[0])
            bounds.max[1] = Math.max(bounds.max[1], objects[i].bounds.max[1])
            bounds.max[2] = Math.max(bounds.max[2], objects[i].bounds.max[2])
        }

        let TLAS = await builders.buildTLAS({ objects, bounds })

        // create the object descriptor buffer
        let objectsBuffer = await builders.packObjects(objects, TLAS.rearrangeBuffer, utilizedMeshOrder)

        sceneGPUState = {
            tlasBVHBuffer : TLAS.bvhBuffer,
            objectsBuffer : objectsBuffer,
            meshBVHBuffer : packedMeshes.bvhBuffer,
            meshTriBuffer : packedMeshes.triBuffer
        }

        const BG_LAYOUT = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }
                }
            ]
        })

        const BG = device.createBindGroup({
            layout: BG_LAYOUT,
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    resource: {
                        buffer: sceneGPUState.tlasBVHBuffer
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    resource: {
                        buffer: sceneGPUState.objectsBuffer
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    resource: {
                        buffer: sceneGPUState.meshBVHBuffer
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    resource: {
                        buffer: sceneGPUState.meshTriBuffer
                    }
                }
            ]
        })

        sceneBindGroupInfo = {
            bindGroup: BG,
            bindGroupLayout: BG_LAYOUT
        }
    }

    function getSceneBindGroupInfo() {
        if (sceneBindGroupInfo == null) console.warn("ERROR in scene::getSceneBindGroupInfo: scene has not been built yet")
        return sceneBindGroupInfo
    }

    function getHitInfoCode(sceneBufferGroupIndex) {
        return /* wgsl */ `
        struct BVHNode {
            aabb_l_min : vec3f,
            l_child :   i32,
            aabb_l_max : vec3f,
                f_1 :   i32,
            aabb_r_min : vec3f,
            r_child :   i32,
            aabb_r_max : vec3f,
                f_2 :   i32
        };

        // additional information is stored in the bottom row of the matrices
        struct Object {
            localToWorld_r_0 : vec4f,
            localToWorld_r_1 : vec4f,
            localToWorld_r_2 : vec4f,
                  bvh_offset :   i32,
                         f_1 :   i32,
                         f_2 :   i32,
                         f_3 :   i32,
            worldToLocal_r_0 : vec4f,
            worldToLocal_r_1 : vec4f,
            worldToLocal_r_2 : vec4f,
                         f_4 :   i32,
                         f_5 :   i32,
                         f_6 :   i32,
                         f_7 :   i32,
        };

        struct Triangle {
            v0 : vec3f,
            v1 : vec3f,
            v2 : vec3f
        };

        @group(${sceneBufferGroupIndex}) @binding(0) var<storage, read_write>    tlas_bvh : array<BVHNode>;
        @group(${sceneBufferGroupIndex}) @binding(1) var<storage, read_write>     objects : array<Object>;
        @group(${sceneBufferGroupIndex}) @binding(2) var<storage, read_write>    mesh_bvh : array<BVHNode>;
        @group(${sceneBufferGroupIndex}) @binding(3) var<storage, read_write>   mesh_tris : array<Triangle>;

        struct TriangleHitInfo {
            normal : vec3f,
              dist : f32
        };

        fn get_triangle_hit_info(o : vec3f, d : vec3f, hit_obj : i32, hit_tri : i32) -> TriangleHitInfo {
            var obj : Object = objects[obj_idx];

            // transform the ray position & direction
            var world_to_local : mat4x4f = mat4x4f(
                vec4f(obj.worldToLocal_r_0.x, obj.worldToLocal_r_1.x, obj.worldToLocal_r_2.x, 0.f),
                vec4f(obj.worldToLocal_r_0.y, obj.worldToLocal_r_1.y, obj.worldToLocal_r_2.y, 0.f),
                vec4f(obj.worldToLocal_r_0.z, obj.worldToLocal_r_1.z, obj.worldToLocal_r_2.z, 0.f),
                vec4f(obj.worldToLocal_r_0.w, obj.worldToLocal_r_1.w, obj.worldToLocal_r_2.w, 1.f),
            );

            var local_o : vec3f = (world_to_local * vec4f(o, 1.f)).xyz;
            var local_d : vec3f = (world_to_local * vec4f(d, 0.f)).xyz;

            var local_hit_info : TriangleHitInfo = helper_intersect_triangle(local_o, local_d, mesh_tris[hit_tri]);

            var local_hit_pos : vec3f = local_o + local_d * local_hit_info.dist;

            var local_to_world : mat4x4f = mat4x4f(
                vec4f(obj.localToWorld_r_0.x, obj.localToWorld_r_1.x, obj.localToWorld_r_2.x, 0.f),
                vec4f(obj.localToWorld_r_0.y, obj.localToWorld_r_1.y, obj.localToWorld_r_2.y, 0.f),
                vec4f(obj.localToWorld_r_0.z, obj.localToWorld_r_1.z, obj.localToWorld_r_2.z, 0.f),
                vec4f(obj.localToWorld_r_0.w, obj.localToWorld_r_1.w, obj.localToWorld_r_2.w, 1.f),
            );

            var world_hit_pos  : vec3f = (local_to_world * vec4f(local_hit_pos, 1.f)).xyz;
            var world_hit_norm : vec3f = (local_to_world * vec4f(local_hit_info.norm, 0.f)).xyz;

            return {
                normalize(world_hit_norm),
                length(world_hit_pos - o)
            };
        }
        
        fn helper_intersect_triangle(o : vec3f, d : vec3f, tri : Triangle) -> TriangleHitInfo {
            var returned : TriangleHitInfo;

            var v0v1 : vec3f = tri.v1 - tri.v0;
            var v0v2 : vec3f = tri.v2 - tri.v0;
            var pvec : vec3f = cross(d, v0v2);

            var   det : f32   = dot(v0v1, pvec);
            var i_det : f32   = 1.f / det;
            var  tvec : vec3f = o - tri.v0;

            var    u : f32   = dot(tvec, pvec) * i_det;
            var qvec : vec3f = cross(tvec, v0v1);
            var    v : f32   = dot(d, qvec) * i_det;

            returned.normal = normalize(cross(v0v1, v0v2));
            returned.dist   = dot(v0v2, qvec) * i_det;

            return returned;
        }
        `
    }

    function getNearestHitCode(sceneBufferGroupIndex) {
        return /* wgsl */ `
        
        struct BVHNode {
            aabb_l_min : vec3f,
            l_child :   i32,
            aabb_l_max : vec3f,
                f_1 :   i32,
            aabb_r_min : vec3f,
            r_child :   i32,
            aabb_r_max : vec3f,
                f_2 :   i32
        };

        // additional information is stored in the bottom row of the matrices
        struct Object {
            localToWorld_r_0 : vec4f,
            localToWorld_r_1 : vec4f,
            localToWorld_r_2 : vec4f,
                  bvh_offset :   i32,
                         f_1 :   i32,
                         f_2 :   i32,
                         f_3 :   i32,
            worldToLocal_r_0 : vec4f,
            worldToLocal_r_1 : vec4f,
            worldToLocal_r_2 : vec4f,
                         f_4 :   i32,
                         f_5 :   i32,
                         f_6 :   i32,
                         f_7 :   i32,
        };

        struct Triangle {
            v0 : vec3f,
            v1 : vec3f,
            v2 : vec3f
        };

        @group(${sceneBufferGroupIndex}) @binding(0) var<storage, read_write>    tlas_bvh : array<BVHNode>;
        @group(${sceneBufferGroupIndex}) @binding(1) var<storage, read_write>     objects : array<Object>;
        @group(${sceneBufferGroupIndex}) @binding(2) var<storage, read_write>    mesh_bvh : array<BVHNode>;
        @group(${sceneBufferGroupIndex}) @binding(3) var<storage, read_write>   mesh_tris : array<Triangle>;

        var<private> stack : array<i32, 32>;

        fn intersect_bvh(o_in : vec3f, d_in : vec3f) -> f32 {
            var o : vec3f = o_in;
            var d : vec3f = d_in;

            var hit_dist : f32 = 1e6f;
            var hit_obj  : i32 = -1;
            
            var stack_ptr : i32 =  0;
            var  node_idx : i32 =  0;
            var   obj_idx : i32 = -1;
            var switch_pt : i32 = -1;

            var iter = 0;

            while (stack_ptr >= 0) {
                iter += 1;
                if (stack_ptr < switch_pt) {
                    // if this is the case, we just left the object

                    // transform the ray position & direction
                    var hit_pos : vec3f = o + d * hit_dist;

                    var obj : Object = objects[obj_idx];
                    var transform_mat : mat4x4f = mat4x4f(
                        vec4f(obj.localToWorld_r_0.x, obj.localToWorld_r_1.x, obj.localToWorld_r_2.x, 0.f),
                        vec4f(obj.localToWorld_r_0.y, obj.localToWorld_r_1.y, obj.localToWorld_r_2.y, 0.f),
                        vec4f(obj.localToWorld_r_0.z, obj.localToWorld_r_1.z, obj.localToWorld_r_2.z, 0.f),
                        vec4f(obj.localToWorld_r_0.w, obj.localToWorld_r_1.w, obj.localToWorld_r_2.w, 1.f),
                    );

                    var new_o : vec3f = (transform_mat * vec4f(o, 1.f)).xyz;
                    var new_h : vec3f = (transform_mat * vec4f(hit_pos, 1.)).xyz;

                    d = new_h - new_o;
                    hit_dist = length(d);
                    d = d / hit_dist;
                    o = new_o;

                    // update other state variables
                    switch_pt = -1;
                    obj_idx   = -1;
                }
                if (node_idx < 0) {
                    if (obj_idx < 0) {
                        // this is an object
                        obj_idx = -(node_idx + 1);
                        var obj : Object = objects[obj_idx];
                        node_idx = obj.bvh_offset;
                        switch_pt = stack_ptr;

                        // transform the ray position & direction
                        var hit_pos : vec3f = o + d * hit_dist;
                        var transform_mat : mat4x4f = mat4x4f(
                            vec4f(obj.worldToLocal_r_0.x, obj.worldToLocal_r_1.x, obj.worldToLocal_r_2.x, 0.f),
                            vec4f(obj.worldToLocal_r_0.y, obj.worldToLocal_r_1.y, obj.worldToLocal_r_2.y, 0.f),
                            vec4f(obj.worldToLocal_r_0.z, obj.worldToLocal_r_1.z, obj.worldToLocal_r_2.z, 0.f),
                            vec4f(obj.worldToLocal_r_0.w, obj.worldToLocal_r_1.w, obj.worldToLocal_r_2.w, 1.f),
                        );

                        var new_o : vec3f = (transform_mat * vec4f(o, 1.f)).xyz;
                        var new_h : vec3f = ((transform_mat * vec4f(hit_pos, 1.f)).xyz);

                        d = new_h - new_o;
                        hit_dist = length(d);
                        d = d / hit_dist;
                        o = new_o;
                    } else {
                        // this is a triangle
                        var   tr : Triangle = mesh_tris[-(node_idx + 1)];
                        var dist :      f32 = helper_tri_intersect(o, d, tr);
                        if (dist > 0.f && dist < hit_dist) {
                            hit_dist = dist;
                        }
                        stack_ptr -= 1;
                        node_idx = stack[stack_ptr];
                    }
                } else {
                    // otherwise, this is an internal BVH node
                    var node : BVHNode;

                    if (obj_idx < 0) {
                        node = tlas_bvh[node_idx];
                    } else {
                        node = mesh_bvh[node_idx];
                    }

                    var l_dist : f32 = helper_aabb_intersect(
                        node.aabb_l_min, 
                        node.aabb_l_max,
                        o, d
                    );
                    var r_dist : f32 = helper_aabb_intersect(
                        node.aabb_r_min,
                        node.aabb_r_max,
                        o, d
                    );
                    var l_valid : bool = l_dist != -1e30f && l_dist < hit_dist;
                    var r_valid : bool = r_dist != -1e30f && r_dist < hit_dist;
                    if (l_valid && r_valid) {
                        var f_idx : i32;
                        var c_idx : i32;

                        if (l_dist < r_dist) {
                            c_idx = node.l_child;
                            f_idx = node.r_child;
                        } else {
                            c_idx = node.r_child;
                            f_idx = node.l_child;
                        }

                        stack[stack_ptr] = f_idx;
                        stack_ptr += 1;
                        node_idx = c_idx;
                    } else
                    if (l_valid) {
                        node_idx = node.l_child;
                    } else 
                    if (r_valid) {
                        node_idx = node.r_child;
                    } else {
                        stack_ptr -= 1;
                        node_idx = stack[stack_ptr];
                    }
                }
            }

            if (obj_idx >= 0) {
                // if this is the case, we just left the object

                // transform the ray position & direction
                var hit_pos : vec3f = o + d * hit_dist;

                var obj : Object = objects[obj_idx];
                var transform_mat : mat4x4f = mat4x4f(
                    vec4f(obj.localToWorld_r_0.x, obj.localToWorld_r_1.x, obj.localToWorld_r_2.x, 0.f),
                    vec4f(obj.localToWorld_r_0.y, obj.localToWorld_r_1.y, obj.localToWorld_r_2.y, 0.f),
                    vec4f(obj.localToWorld_r_0.z, obj.localToWorld_r_1.z, obj.localToWorld_r_2.z, 0.f),
                    vec4f(obj.localToWorld_r_0.w, obj.localToWorld_r_1.w, obj.localToWorld_r_2.w, 1.f),
                );

                var new_o : vec3f = (transform_mat * vec4f(o, 1.f)).xyz;
                var new_h : vec3f = ((transform_mat * vec4f(hit_pos, 1.)).xyz);

                d = new_h - new_o;
                hit_dist = length(d);
                d = d / hit_dist;

                // update other state variables
                switch_pt = -1;
                obj_idx   = -1;
            }

            return hit_dist;
        }

        // from: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
        fn helper_tri_intersect(o : vec3f, d : vec3f, tri : Triangle) -> f32 {
            var v0v1 : vec3f = tri.v1 - tri.v0;
            var v0v2 : vec3f = tri.v2 - tri.v0;
            var pvec : vec3f = cross(d, v0v2);

            var  det : f32 = dot(v0v1, pvec);

            if (abs(det) < 1e-10) {
                return -1.f;
            }

            var i_det : f32   = 1.f / det;
            var  tvec : vec3f = o - tri.v0;

            var u : f32 = dot(tvec, pvec) * i_det;
            
            if (u < 0.f || u > 1.f) {
                return -1.f;
            }

            var qvec : vec3f = cross(tvec, v0v1);

            var v : f32 = dot(d, qvec) * i_det;
            if (v < 0.f || u + v > 1.f) {
                return -1.f;
            }

            return dot(v0v2, qvec)  * i_det;
        }

        fn helper_aabb_intersect(low : vec3f, high : vec3f, o : vec3f, d : vec3f) -> f32 {
            var iDir = 1. / d;
            var f = (high - o) * iDir; var n = (low - o) * iDir;
            var tmax = max(f, n); var tmin = min(f, n);
            var t0 = max(tmin.x, max(tmin.y, tmin.z));
            var t1 = min(tmax.x, min(tmax.y, tmax.z));
            return select(-1e30, select(t0, -1e30, t1 < 0.), t1 >= t0);
        }`
    }

    function registerMesh(mesh) {
        meshes.push({ mesh: mesh })
        return meshes.length - 1
    }

    function instanceMesh(meshID, position, rotation, scale) {
        if (meshID >= meshes.length || meshID < 0) {
            console.warn(`ERROR in scene::instanceMesh: mesh with ID ${meshID} does not exist`)
            return
        }
        objects.push({ type: "mesh", meshID, transform: { position, rotation, scale } })
    }

    function addSphere(position, radius) {
        objects.push({ type: "sphere", position, radius })
    }
}