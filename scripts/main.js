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
    
    for (var i = 0; i < 50; i++) {
        scene.instanceMesh(Math.floor(2), [
            6 * Math.random(),
            6 * Math.random(),
            6 * Math.random()
        ], [
            Math.random() * 6.28, Math.random() * 6.28, Math.random() * 6.28
        ], [
            1, 1, 1
        ])
    }

    await scene.build()

    /*const debug = initDebug(device, document.querySelector("#canvas"), scene)

    const t1 = Date.now()
    console.log(t1 - t0)

    for (var x = 0; x < 1; x++) {
        const ta = Date.now()
        await debug.draw()
        const tb = Date.now()
        console.log(tb - ta)
    }

    return*/

    const pt = initPathTracer({ 
        device, scene,
        image: {
            width: 1000, height: 1000
        },
        camera: {
            lookAt: [0, 0, 0],
            position: [7, 7, 7],
            fov: 60
        }
    })

    const display = initDisplay({
        device, 
        canvas: document.querySelector("#canvas"),
        image: {
            buffer: pt.getImageBuffer(),
            width: 1000,
            height: 1000
        }
    })

    async function frame() {
        for (var i = 0; i < 1; i++) {
            await pt.step()
        }

        await display.draw()

        console.log("hello")

        window.requestAnimationFrame(frame)
    }

    frame()

    /*for (var x = 0; x < 1000; x++) {
        await pt.step()
    }

    display.draw()*/
}