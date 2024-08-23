

function initBuilders(device) {
    const radixSortKernel = initRadixSortKernel(device)
    const radixTreeKernel = initRadixTreeKernel(device)
    const aabb_ZidxKernel = initAABB_ZidxKernel(device)
    const bvhUpPassKernel = initBVHUpPassKernel(device)
    const rearrangeKernel = initRearrangeKernel(device)
    const packTMeshKernel = initPackTMeshKernel(device)

    return { buildMeshBVH, packMeshes }

    async function packMeshes(meshes, utilizedMeshes) {
        let bvhs = []
        for (var x in utilizedMeshes) {
            bvhs.push(meshes[x].bvh)
        }
        const res = await packTMeshKernel.execute(bvhs)
        return {
            bvhBuffer: res.PACKED_BVH_BUFFER,
            triBuffer: res.PACKED_TRI_BUFFER,
            offsets  : res.OFFSETS
        }
    }

    async function buildMeshBVH(mesh) {
        // parse the OBJ file to get the list of triangles, mesh bounds, etc.
        const triangles = parseObj(mesh.file)

        // create GPU triangle buffer and copy values to it
        const I_TRIANGE_BUFFER = device.createBuffer({
            size: triangles.numTriangles * 48,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        })

        new Float32Array(I_TRIANGE_BUFFER.getMappedRange()).set(triangles.triangleArray)
        I_TRIANGE_BUFFER.unmap()

        // compute AABB and morton code for each triangle
        const { AABB_BUFFER, Z_IDX_BUFFER } = await aabb_ZidxKernel.execute(
            I_TRIANGE_BUFFER,
            triangles.numTriangles,
            triangles.bounds
        )

        // sort the morton code buffer and store how indices change
        const { IDX_BUFFER } = await radixSortKernel.execute(
            Z_IDX_BUFFER,
            triangles.numTriangles
        )

        // compute the radix tree over the morton codes
        const { PARENT_BUFFER } = await radixTreeKernel.execute(
            Z_IDX_BUFFER,
            triangles.numTriangles
        )

        // combine all information from previous passes into BVH
        const { BVH_BUFFER } = await bvhUpPassKernel.execute(
            IDX_BUFFER,
            AABB_BUFFER,
            PARENT_BUFFER,
            triangles.numTriangles
        )

        // rearrange the triangles
        const { O_TRIANGLE_BUFFER } = await rearrangeKernel.execute(
            I_TRIANGE_BUFFER,
            IDX_BUFFER,
            triangles.numTriangles
        )

        // free all buffers that are not input/output
        AABB_BUFFER.destroy()
        Z_IDX_BUFFER.destroy()
        PARENT_BUFFER.destroy()
        IDX_BUFFER.destroy()
        I_TRIANGE_BUFFER.destroy()

        return { 
            bvhBuffer: BVH_BUFFER, 
            triangleBuffer: O_TRIANGLE_BUFFER,
            numTriangles: triangles.numTriangles,
            bounds: triangles.bounds
        }
    }
}

// old code
function initBVHBuild(device) {

    // initialize all other shaders
    const radixSortKernel = initRadixSortKernel(device)
    const radixTreeKernel = initRadixTreeKernel(device)
    const aabb_ZidxKernel = initAABB_ZidxKernel(device)
    const bvhUpPassKernel = initBVHUpPassKernel(device)
    const rearrangeKernel = initRearrangeKernel(device)

    return { build }

    async function build(TRI_ARRAY, NUM_TRIS, MODEL_BOUNDS) {
        // create GPU triangle buffer and copy values to it
        const I_TRIANGE_BUFFER = device.createBuffer({
            size: NUM_TRIS * 48,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        })

        new Float32Array(I_TRIANGE_BUFFER.getMappedRange()).set(TRI_ARRAY)
        I_TRIANGE_BUFFER.unmap()
        
        // compute AABB and morton code for each triangle
        const { AABB_BUFFER, Z_IDX_BUFFER } = await aabb_ZidxKernel.execute(
            I_TRIANGE_BUFFER,
            NUM_TRIS,
            MODEL_BOUNDS
        )

        // sort the morton code buffer and store how indices change
        const { IDX_BUFFER } = await radixSortKernel.execute(
            Z_IDX_BUFFER,
            NUM_TRIS
        )

        // compute the radix tree over the morton codes
        const { PARENT_BUFFER } = await radixTreeKernel.execute(
            Z_IDX_BUFFER,
            NUM_TRIS
        )

        // combine all information from previous passes into BVH
        const { BVH_BUFFER } = await bvhUpPassKernel.execute(
            IDX_BUFFER,
            AABB_BUFFER,
            PARENT_BUFFER,
            NUM_TRIS
        )

        // rearrange the triangles
        const { O_TRIANGLE_BUFFER } = await rearrangeKernel.execute(
            I_TRIANGE_BUFFER,
            IDX_BUFFER,
            NUM_TRIS
        )

        // free all buffers that are not input/output
        AABB_BUFFER.destroy()
        Z_IDX_BUFFER.destroy()
        PARENT_BUFFER.destroy()
        IDX_BUFFER.destroy()
        I_TRIANGE_BUFFER.destroy()

        return { BVH_BUFFER, O_TRIANGLE_BUFFER }
    }
}