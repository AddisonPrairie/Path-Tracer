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

    scene.registerMesh({ file: cowFile })
    scene.registerMesh({ file: bunnyFile })
    scene.registerMesh({ file: buddhaFile })
    
    for (var i = 0; i < 20; i++) {
        scene.instanceMesh(Math.floor(0), [
            6 * Math.random(),
            6 * Math.random(),
            6 * Math.random(),
        ], [
            Math.PI * .5, 0, Math.PI * 1.25
        ], [
            .5, .5, .5
        ])
    }

    await scene.build()

    const trace = scene.getTraceKernels(1)

    const debug = initDebug(device, document.querySelector("#canvas"), trace)

    const t1 = Date.now()
    console.log(t1 - t0)

    for (var x = 0; x < 10; x++) {
        const ta = Date.now()
        await debug.draw()
        const tb = Date.now()
        console.log(tb - ta)
    }
}