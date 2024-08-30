

window.onload = async () => {
    // initialize WebGPU
    const adapter = await navigator.gpu?.requestAdapter()
    const device  = await adapter?.requestDevice()

    if (!device) {
        alert("browser does not support WebGPU!")
        return
    }

    let cowFile   = await fetch("media/cow.obj"  ).then(f => f.text())
    let floorFile = await fetch("media/floor.obj").then(f => f.text())

    const scene = initScene(device)

    scene.registerMesh({ file: cowFile })
    scene.registerMesh({ file: floorFile })
    
    //scene.instanceMesh(1, [6, 1, 0], [0, 0, 0], [1, 1, 1])
    //scene.instanceMesh(0, [6, 0, 6], [0.5, 0, 0], [1, 1, 1])
    for (var i = 0; i < 200; i++) {
        scene.instanceMesh(0, [
            6 * Math.random(),
            6 * Math.random(),
            6 * Math.random(),
        ], [
            Math.random() * 6.28,
            Math.random() * 6.28,
            Math.random() * 6.28,
        ], [
            .1, .1, .1
        ])
    }

    await scene.build()

    const trace = scene.getTraceKernels()

    const debug = initDebug(device, document.querySelector("#canvas"), trace)

    debug.draw()
}