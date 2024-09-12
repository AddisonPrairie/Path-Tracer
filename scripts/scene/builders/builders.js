function initBuilders(device) {
    const radixSortKernel = initRadixSortKernel(device)
    const radixTreeKernel = initRadixTreeKernel(device)
    const aabb_ZidxKernel = initAABB_ZidxKernel(device)
    const aabb2ZidxKernel = initAABB2ZidxKernel(device)
    const bvhUpPassKernel = initBVHUpPassKernel(device)
    const rearrangeKernel = initRearrangeKernel(device)
    const packTMeshKernel = initPackTMeshKernel(device)

    return { buildMeshBVH, packMeshes, buildTLAS, packObjects }

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

    async function packObjects(objects, rearrangeBuffer, utilizedMeshOrder) {
        let rearrangeOrder = new Int32Array(await readBackBuffer(device, rearrangeBuffer))

        let objectsBuf = new ArrayBuffer(objects.length * 128)
        {
            let DV = new DataView(objectsBuf)
            for (var i = 0; i < objects.length; i++) {
                // local -> world matrix
                DV.setFloat32(128 * i + 0 , objects[rearrangeOrder[i]].transformMatrices.localToWorld[ 0], true)
                DV.setFloat32(128 * i + 4 , objects[rearrangeOrder[i]].transformMatrices.localToWorld[ 4], true)
                DV.setFloat32(128 * i + 8 , objects[rearrangeOrder[i]].transformMatrices.localToWorld[ 8], true)
                DV.setFloat32(128 * i + 12, objects[rearrangeOrder[i]].transformMatrices.localToWorld[12], true)
                DV.setFloat32(128 * i + 16, objects[rearrangeOrder[i]].transformMatrices.localToWorld[ 1], true)
                DV.setFloat32(128 * i + 20, objects[rearrangeOrder[i]].transformMatrices.localToWorld[ 5], true)
                DV.setFloat32(128 * i + 24, objects[rearrangeOrder[i]].transformMatrices.localToWorld[ 9], true)
                DV.setFloat32(128 * i + 28, objects[rearrangeOrder[i]].transformMatrices.localToWorld[13], true)
                DV.setFloat32(128 * i + 32, objects[rearrangeOrder[i]].transformMatrices.localToWorld[ 2], true)
                DV.setFloat32(128 * i + 36, objects[rearrangeOrder[i]].transformMatrices.localToWorld[ 6], true)
                DV.setFloat32(128 * i + 40, objects[rearrangeOrder[i]].transformMatrices.localToWorld[10], true)
                DV.setFloat32(128 * i + 44, objects[rearrangeOrder[i]].transformMatrices.localToWorld[14], true)

                // world -> local matrix
                DV.setFloat32(128 * i +  64, objects[rearrangeOrder[i]].transformMatrices.worldToLocal[ 0], true)
                DV.setFloat32(128 * i +  68, objects[rearrangeOrder[i]].transformMatrices.worldToLocal[ 4], true)
                DV.setFloat32(128 * i +  72, objects[rearrangeOrder[i]].transformMatrices.worldToLocal[ 8], true)
                DV.setFloat32(128 * i +  76, objects[rearrangeOrder[i]].transformMatrices.worldToLocal[12], true)
                DV.setFloat32(128 * i +  80, objects[rearrangeOrder[i]].transformMatrices.worldToLocal[ 1], true)
                DV.setFloat32(128 * i +  84, objects[rearrangeOrder[i]].transformMatrices.worldToLocal[ 5], true)
                DV.setFloat32(128 * i +  88, objects[rearrangeOrder[i]].transformMatrices.worldToLocal[ 9], true)
                DV.setFloat32(128 * i +  92, objects[rearrangeOrder[i]].transformMatrices.worldToLocal[13], true)
                DV.setFloat32(128 * i +  96, objects[rearrangeOrder[i]].transformMatrices.worldToLocal[ 2], true)
                DV.setFloat32(128 * i + 100, objects[rearrangeOrder[i]].transformMatrices.worldToLocal[ 6], true)
                DV.setFloat32(128 * i + 104, objects[rearrangeOrder[i]].transformMatrices.worldToLocal[10], true)
                DV.setFloat32(128 * i + 108, objects[rearrangeOrder[i]].transformMatrices.worldToLocal[14], true)

                // additional info
                DV.setInt32  (128 * i +  48, utilizedMeshOrder[objects[rearrangeOrder[i]].meshID].triangleOffset, true)
            }
        }
        
        const BUFFER = device.createBuffer({
            size: objects.length * 128,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true
        })

        new Int32Array(BUFFER.getMappedRange()).set(new Int32Array(objectsBuf))
        BUFFER.unmap()

        return BUFFER

        // util function to read back rearrange buffer
        async function readBackBuffer(device, buffer) {
            const readBuffer = device.createBuffer({
                size: buffer.size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
            })
            const CE = device.createCommandEncoder()
            CE.copyBufferToBuffer(buffer, 0, readBuffer, 0, buffer.size)
            device.queue.submit([CE.finish()])
            await readBuffer.mapAsync(GPUMapMode.READ)
            const ret = new ArrayBuffer(buffer.size)
            new Int32Array(ret).set(new Int32Array(readBuffer.getMappedRange()))
            readBuffer.destroy()
            return ret
        }
    }

    async function buildTLAS(scene) {
        // create the AABB buffer and send it to the GPU
        let aabbs = new Float32Array(scene.objects.length * 8)
        for (var i = 0; i < scene.objects.length; i++) {
            aabbs[8 * i + 0] = scene.objects[i].bounds.min[0]
            aabbs[8 * i + 1] = scene.objects[i].bounds.min[1]
            aabbs[8 * i + 2] = scene.objects[i].bounds.min[2]
            aabbs[8 * i + 4] = scene.objects[i].bounds.max[0]
            aabbs[8 * i + 5] = scene.objects[i].bounds.max[1]
            aabbs[8 * i + 6] = scene.objects[i].bounds.max[2]
        }

        const AABB_BUFFER = device.createBuffer({
            size: scene.objects.length * 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        })

        new Float32Array(AABB_BUFFER.getMappedRange()).set(aabbs)
        AABB_BUFFER.unmap()

        const { Z_IDX_BUFFER } = await aabb2ZidxKernel.execute(AABB_BUFFER, scene.objects.length, scene.bounds)

        // sort the morton code buffer and store how indices change
        const { IDX_BUFFER } = await radixSortKernel.execute(
            Z_IDX_BUFFER,
            scene.objects.length
        )

        // compute the radix tree over the morton codes
        const { PARENT_BUFFER } = await radixTreeKernel.execute(
            Z_IDX_BUFFER,
            scene.objects.length
        )

        // combine all information from previous passes into BVH
        const { BVH_BUFFER } = await bvhUpPassKernel.execute(
            IDX_BUFFER,
            AABB_BUFFER,
            PARENT_BUFFER,
            scene.objects.length
        )

        // free all buffers that are not input/output
        AABB_BUFFER.destroy()
        Z_IDX_BUFFER.destroy()
        PARENT_BUFFER.destroy()

        return { 
            bvhBuffer: BVH_BUFFER,
            rearrangeBuffer: IDX_BUFFER,
            numObjects: scene.objects.numObjects,
            bounds: scene.bounds
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