

window.onload = async () => {
    // initialize WebGPU
    const adapter = await navigator.gpu?.requestAdapter()
    const device  = await adapter?.requestDevice()

    if (!device) {
        alert("browser does not support WebGPU!")
        return
    }

    let cowFile = await fetch("media/cow.obj").then(f => f.text())

    const scene = initScene(device)

    scene.registerMesh({file: cowFile})
    scene.registerMesh({file: cowFile})
    scene.registerMesh({file: cowFile})

    scene.instanceMesh(0, [0, 0, 0], [0, 0, 0], [1, 1, 1])
    scene.instanceMesh(2, [0, 0, 0], [0, 0, 0], [1, 1, 1])

    await scene.build()

    console.log(scene)
}