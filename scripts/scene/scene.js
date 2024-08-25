

function initScene(device) {

    const builders = initBuilders(device)

    // externally editable state
    let meshes  = []
    let objects = []

    // internal state
    let packedMeshes = null

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
        packedMeshes = await builders.packMeshes(meshes, utilizedMeshOrder)

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
    }

    function getTraceKernels() {

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

