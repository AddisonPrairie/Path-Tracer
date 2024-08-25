

function initScene(device) {

    const builders = initBuilders(device)

    // externally editable state
    let meshes  = []
    let objects = []

    // internal state
    let sceneGPUState = {}

    return { registerMesh, instanceMesh, build, getTraceKernels }

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
    }

    function getTraceKernels() {
        if (sceneGPUState == null) {
            console.warn("ERROR in scene::getTraceKernels: should not call before scene is built!")
            return
        }

        const SRC = /* wgsl */ `

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
            localToWorld : mat4x4f,
            worldToLocal : mat4x4f
        };

        struct Triangle {
            v0 : vec3f,
            v1 : vec3f,
            v2 : vec3f
        };

        @group(0) @binding(0) var<storage, read_write>    tlas_bvh : array<BVHNode>;
        @group(0) @binding(1) var<storage, read_write>     objects : array<Object>;
        @group(0) @binding(2) var<storage, read_write>    mesh_bvh : array<BVHNode>;
        @group(0) @binding(3) var<storage, read_write>   mesh_tris : array<Triangle>;
        
        struct RayHit {
            dist :   f32,
            norm : vec3f,
        };

        var<private> stack : array<i32, 32>;

        fn intersect_bvh(o_in : vec3f, d_in : vec3f) -> RayHit {
            var o : vec3f = o_in;
            var d : vec3f = d_in;

            var hit_dist : f32 = 1e30f;
            var hit_obj  : i32 = -1;
            
            var stack_ptr : i32 =  0;
            var  node_idx : i32 =  0;
            var   obj_idx : i32 = -1;
            var switch_pt : i32 = -1;

            while (stack_ptr >= 0) {
                if (stack_ptr < switch_pt) {
                    // if this is the case, we just left the object
                    switch_pt = -1;
                    obj_idx   = -1;
                }
                if (node_idx < 0) {
                    if (obj_idx < 0) {
                        // this is an object
                        obj_idx = -(node_idx + 1);
                        var obj : Object = objects[obj_idx];
                        node_idx = get_object_bvh_offset(obj);
                    } else {
                        // this is a triangle
                        var   tr : Triangle = mesh_tris[-(node_idx + 1)];
                        var dist :      f32 = tri_intersect(o, d, tr);
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

                    var l_dist : f32 = aabb_intersect(
                        node.aabb_l_min, 
                        node.aabb_l_max,
                        o, d
                    );
                    var r_dist : f32 = aabb_intersect(
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
        }

        fn get_object_bvh_offset(obj : Object) -> i32 {
            return std::bitcast<i32>(obj.localToWorld.a);
        }

        // from: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
        fn tri_intersect(o : vec3f, d : vec3f, tri : Triangle) -> vec4f {
            var v0v1 : vec3f = tri.v1 - tri.v0;
            var v0v2 : vec3f = tri.v2 - tri.v0;
            var pvec : vec3f = cross(d, v0v2);

            var  det : f32 = dot(v0v1, pvec);

            if (abs(det) < 1e-10) {
                return vec4f(-1.f);
            }

            var i_det : f32   = 1.f / det;
            var  tvec : vec3f = o - tri.v0;

            var u : f32 = dot(tvec, pvec) * i_det;
            
            if (u < 0.f || u > 1.f) {
                return vec4f(-1.f);
            }

            var qvec : vec3f = cross(tvec, v0v1);

            var v : f32 = dot(d, qvec) * i_det;
            if (v < 0.f || u + v > 1.f) {
                return vec4f(-1.f);
            }

            return vec4f(
                normalize(cross(v0v1, v0v2)),
                dot(v0v2, qvec) * i_det
            );
        }

        fn aabb_intersect(low : vec3f, high : vec3f, o : vec3f, d : vec3f) -> f32 {
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

