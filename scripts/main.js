

window.onload = async () => {
    // initialize WebGPU
    const adapter = await navigator.gpu?.requestAdapter()
    const device  = await adapter?.requestDevice()

    if (!device) {
        alert("browser does not support WebGPU!")
        return
    }

    let cowFile    = await fetch("media/cow.obj"  ).then(f => f.text())
    let floorFile  = await fetch("media/floor.obj").then(f => f.text())
    let bunnyFile  = await fetch("media/bunny.obj").then(f => f.text())
    let buddhaFile = await fetch("media/buddha.obj").then(f => f.text())

    const scene = initScene(device)

    const t0 = Date.now()

    //scene.registerMesh({ file: cowFile })
    scene.registerMesh({ file: bunnyFile })
    scene.registerMesh({ file: buddhaFile })
    
    for (var i = 0; i < 200; i++) {
        scene.instanceMesh(Math.floor(Math.random() * 2), [
            6 * Math.random(),
            6 * Math.random(),
            6 * Math.random(),
        ], [
            Math.random() * 6.28,
            Math.random() * 6.28,
            Math.random() * 6.28,
        ], [
            .5, .5, .5
        ])
    }

    await scene.build()

    const trace = scene.getTraceKernels()

    const debug = initDebug(device, document.querySelector("#canvas"), trace)

    await debug.draw()

    const t1 = Date.now()

    console.log(t1 - t0)
}