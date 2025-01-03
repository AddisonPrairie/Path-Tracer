function initScene(device) {

    const builders = initBuilders(device)

    // externally editable state
    let meshes = []
    let objects = []
    let lights = []
    let materials = []

    // internal state
    let sceneGPUState = null
    let sceneBindGroupInfo = null
    let materialBindGroupInfo = null

    let loadedCoreMeshes = {
        "square" : null,
    }

    return { 
        registerMesh, 
        instanceMesh, 
        build, 
        addLight, 
        addMaterial, 
        getSceneBindGroupInfo, 
        getMaterialBindGroupInfo, 
        getLightCount,
        setEnvironmentLight,
        kernels: { 
            getNearestHitCode, 
            getAnyHitCode, 
            getHitInfoCode 
        } 
    }

    async function build() {
        {
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
                bindGroupLayout: BG_LAYOUT,
            }
        }

        {
            const BG_LAYOUT = device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "uniform"
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "uniform"
                        }
                    }
                ]
            })

            const materialBuffer = device.createBuffer({
                size: 64 * 128,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            })

            const lightBuffer = device.createBuffer({
                size: 128 * 128,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            })

            function copyIntoBuffer(source, target, targetOffset) {
                const targetView = new Uint8Array(target)
                const sourceView = new Uint8Array(source)

                for (var x = 0; x < sourceView.length; x++) {
                    targetView[targetOffset + x] = sourceView[x]
                }
            }

            {// put together the material buffer
                const cpuMaterialBuffer = new ArrayBuffer(materials.length * 64)

                for (var x = 0; x < materials.length; x++) {
                    copyIntoBuffer(materials[x].buffer, cpuMaterialBuffer, x * 64)
                }

                device.queue.writeBuffer(materialBuffer, 0, cpuMaterialBuffer, 0)
            }

            {// put together the light buffer
                const cpuLightBuffer = new ArrayBuffer(lights.length * 128)

                for (var x = 0; x < lights.length; x++) {
                    copyIntoBuffer(lights[x].buffer, cpuLightBuffer, x * 128)
                }

                device.queue.writeBuffer(lightBuffer, 0, cpuLightBuffer, 0)
            }

            const BG = device.createBindGroup({
                layout: BG_LAYOUT,
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: materialBuffer
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: lightBuffer
                        }
                    }
                ]
            })

            materialBindGroupInfo = {
                bindGroupLayout: BG_LAYOUT,
                bindGroup: BG
            }
        }
    }

    function addMaterial(type, descriptor) {
        switch (type) {
            case "lambert_diffuse":
                {
                    let r = descriptor.color && descriptor.color.r ? descriptor.color.r : 0.
                    let g = descriptor.color && descriptor.color.g ? descriptor.color.g : 0.
                    let b = descriptor.color && descriptor.color.b ? descriptor.color.b : 0.


                    const buffer = new ArrayBuffer(64)
                    const dataView = new DataView(buffer)

                    dataView.setInt32  (0 , 1, true)
                    dataView.setFloat32(4 , r, true)
                    dataView.setFloat32(8 , g, true)
                    dataView.setFloat32(12, b, true) 

                    let idx = materials.length
                    materials[idx] = { buffer }
                    return idx
                }
            case "mirror":
                {
                    let r = descriptor.color && descriptor.color.r ? descriptor.color.r : 0.
                    let g = descriptor.color && descriptor.color.g ? descriptor.color.g : 0.
                    let b = descriptor.color && descriptor.color.b ? descriptor.color.b : 0.


                    const buffer = new ArrayBuffer(64)
                    const dataView = new DataView(buffer)

                    dataView.setInt32  (0 , 2, true)
                    dataView.setFloat32(4 , r, true)
                    dataView.setFloat32(8 , g, true)
                    dataView.setFloat32(12, b, true)

                    let idx = materials.length
                    materials[idx] = { buffer }
                    return idx
                }
            case "ggx_smith":
                {
                    let r = descriptor.color && descriptor.color.r ? descriptor.color.r : 0.
                    let g = descriptor.color && descriptor.color.g ? descriptor.color.g : 0.
                    let b = descriptor.color && descriptor.color.b ? descriptor.color.b : 0.
                    let roughness = descriptor.roughness ? Math.min(Math.max(.0001, descriptor.roughness), 1.) : .1

                    const buffer = new ArrayBuffer(64)
                    const dataView = new DataView(buffer)

                    dataView.setInt32  (0 , 3, true)
                    dataView.setFloat32(4 , r, true)
                    dataView.setFloat32(8 , g, true)
                    dataView.setFloat32(12, b, true)
                    dataView.setFloat32(16, roughness, true)

                    let idx = materials.length
                    materials[idx] = { buffer }
                    return idx
                }
            case "emissive":
                {
                    let r = descriptor.le && descriptor.le.r ? descriptor.le.r : 0.
                    let g = descriptor.le && descriptor.le.g ? descriptor.le.g : 0.
                    let b = descriptor.le && descriptor.le.b ? descriptor.le.b : 0.

                    const buffer = new ArrayBuffer(64)
                    const dataView = new DataView(buffer)

                    dataView.setInt32  (0 , 4, true)
                    dataView.setFloat32(4 , r, true)
                    dataView.setFloat32(8 , g, true)
                    dataView.setFloat32(12, b, true)

                    let idx = materials.length
                    materials[idx] = { buffer }
                    return idx
                }
            default:
                console.error("ERROR in scene::addMaterial: unknown light type [ ", type, " ]") 
        }
    }

    function addLight(type, descriptor) {
        switch (type) {
            case "rectangle":
                let position = [
                    descriptor.position && descriptor.position.x ? descriptor.position.x : 0.,
                    descriptor.position && descriptor.position.y ? descriptor.position.y : 0.,
                    descriptor.position && descriptor.position.z ? descriptor.position.z : 0.
                ]
                let target = [
                    descriptor.target && descriptor.target.x != null ? descriptor.target.x : position[0],
                    descriptor.target && descriptor.target.y != null ? descriptor.target.y : position[1],
                    descriptor.target && descriptor.target.z != null ? descriptor.target.z : position[2]
                ]
                let forward = [
                    target[0] - position[0], 
                    target[1] - position[1], 
                    target[2] - position[2]
                ]
                let length = Math.sqrt(forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2])
                if (length == 0.) {
                    forward = [0., 0., -1.]
                } else {
                    forward[0] /= length
                    forward[1] /= length
                    forward[2] /= length
                }

                let zProjLength = Math.sqrt(forward[0] * forward[0] + forward[1] * forward[1])

                let xTheta = Math.acos(-forward[2])
                let zTheta = forward[0] < 0. ? Math.acos(forward[1] / zProjLength) : -Math.acos(forward[1] / zProjLength)

                let right = [
                    Math.cos(zTheta - Math.PI * .5),
                    Math.sin(zTheta - Math.PI * .5),
                    0.
                ]

                if (Math.abs(forward[2] + 1.) < 1e-4) {
                    xTheta = 0.
                    zTheta = 0.

                    right = [1, 0, 0]
                }

                let scale = [
                    descriptor.scale && descriptor.scale.x && descriptor.scale.x != 0. ? descriptor.scale.x : 1.,
                    descriptor.scale && descriptor.scale.y && descriptor.scale.y != 0. ? descriptor.scale.y : 1.,
                ]

                if (loadedCoreMeshes.square == null) {
                    loadedCoreMeshes.square = registerMesh({ file: getSquareMesh() })
                }

                let le = [
                    descriptor.le && descriptor.le.r ? descriptor.le.r : 0.,
                    descriptor.le && descriptor.le.g ? descriptor.le.g : 0.,
                    descriptor.le && descriptor.le.b ? descriptor.le.b : 0.
                ]

                let matIndex = addMaterial("emissive", {le: {r: le[0], g: le[1], b: le[2]}})
                instanceMesh(loadedCoreMeshes.square, position, [xTheta, 0, zTheta], [scale[0], scale[1], 1.], matIndex)

                {
                    const buffer = new ArrayBuffer(128)
                    const dataView = new DataView(buffer)

                    dataView.setInt32  (0 ,     1, true) // load light type
                    dataView.setFloat32(4 , le[0], true) // load light LE
                    dataView.setFloat32(8 , le[1], true)
                    dataView.setFloat32(12, le[2], true) 

                    dataView.setFloat32(16, position[0], true) // load light position
                    dataView.setFloat32(20, position[1], true)
                    dataView.setFloat32(24, position[2], true)

                    dataView.setFloat32(32, forward[0], true) // load light forward
                    dataView.setFloat32(36, forward[1], true)
                    dataView.setFloat32(40, forward[2], true)

                    dataView.setFloat32(48, right[0], true) // load light right
                    dataView.setFloat32(52, right[1], true)
                    dataView.setFloat32(56, right[2], true)

                    dataView.setFloat32(64, scale[0], true) // load scale
                    dataView.setFloat32(68, scale[1], true)

                    let idx = lights.length
                    lights[idx] = { buffer }
                    return idx
                }
            default:
                console.error("ERROR in scene::addLight: unknown light type [ ", type, " ]") 
        }
    }

    function setEnvironmentLight(type, info) {

    }

    function getSceneBindGroupInfo() {
        if (sceneBindGroupInfo == null) console.warn("ERROR in scene::getSceneBindGroupInfo: scene has not been built yet")
        return sceneBindGroupInfo
    }

    function getMaterialBindGroupInfo() {
        if (materialBindGroupInfo == null) console.warn("ERROR in scene::getMaterialBindGroupInfo: scene has not been built yet")
        return materialBindGroupInfo
    }

    function getLightCount() {
        return lights.length
    }

    function getHitInfoCode(sceneBufferGroupIndex, noDuplicate) {
        return /* wgsl */ `

        ${
            noDuplicate ?  "" : 
            /* wgsl */ `struct BVHNode {
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
                        material :   i32,
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
        `}

        struct TriangleHitInfo {
            normal : vec3f,
              dist : f32
        };

        fn get_triangle_hit_info(o : vec3f, d : vec3f, hit_obj : i32, hit_tri : i32) -> TriangleHitInfo {
            var obj : Object = objects[hit_obj];

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
            var world_hit_norm : vec3f = (local_to_world * vec4f(local_hit_info.normal, 0.f)).xyz;

            var returned : TriangleHitInfo;
            returned.normal = normalize(world_hit_norm);
            returned.dist   = length(world_hit_pos - o);

            return returned;
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

    function getAnyHitCode(sceneBufferGroupIndex) {
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
                    material :   i32,
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

        struct BVHHitResult {
            hit_obj : i32,
            hit_tri : i32,
            hit_dis : f32
        };

        fn intersect_bvh_any(o_in : vec3f, d_in : vec3f, max_dist : f32) -> bool {
            var o : vec3f = o_in;
            var d : vec3f = d_in;

            var hit_dist : f32 = max_dist;
            var hit_obj  : i32 = -1;
            var hit_tri  : i32 = -1;
            
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
                            /*hit_dist = dist;
                            hit_tri = -(node_idx + 1);
                            hit_obj = obj_idx;*/
                            return true;
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

            return false;
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
                    material :   i32,
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

        struct BVHHitResult {
            hit_obj : i32,
            hit_tri : i32,
            hit_dis : f32
        };

        fn intersect_bvh(o_in : vec3f, d_in : vec3f) -> BVHHitResult {
            var o : vec3f = o_in;
            var d : vec3f = d_in;

            var hit_dist : f32 = 1e6f;
            var hit_obj  : i32 = -1;
            var hit_tri  : i32 = -1;
            
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
                            hit_tri = -(node_idx + 1);
                            hit_obj = obj_idx;
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

            var returned : BVHHitResult;

            returned.hit_obj = hit_obj;
            returned.hit_tri = hit_tri;
            returned.hit_dis = hit_dist;

            return returned;
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

    function instanceMesh(meshID, position, rotation, scale, material) {
        if (meshID >= meshes.length || meshID < 0) {
            console.warn(`ERROR in scene::instanceMesh: mesh with ID ${meshID} does not exist`)
            return
        }
        objects.push({ type: "mesh", meshID, material, transform: { position, rotation, scale } })
    }

    function addSphere(position, radius) {
        objects.push({ type: "sphere", position, radius })
    }
}