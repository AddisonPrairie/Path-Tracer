

function initScene(device) {


    const builders = initBuilders(device)

    let meshes  = []
    let objects = []

    return { registerMesh, instanceMesh, build, getTraceKernels }

    async function build() {
        // build BVH's for all meshes included in the scene
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

        const packedMeshes = await builders.packMeshes(meshes, utilizedMeshOrder)
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
        objects.push({ type: "mesh", meshID, position, rotation, scale })
    }

    function addSphere(position, radius) {
        objects.push({ type: "sphere", position, radius })
    }
}