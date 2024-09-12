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
        scene.instanceMesh(Math.floor(0), [
            6 * Math.random(),
            6 * Math.random(),
            6 * Math.random()
        ], [
            Math.random() * 6.28, Math.random() * 6.28, Math.random() * 6.28
        ], [
            .5, .5, .5
        ])
    }

    await scene.build()

    const pt = initPathTracer({ 
        device, scene,
        image: {
            width: 1000, height: 1000
        },
        camera: {
            lookAt: [0, 0, 0],
            position: [10, 10, 10],
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

    let maxCount = 0

    async function frame() {

        if (maxCount++ > 1_000) return

        for (var i = 0; i < 1; i++) {
            const ta = Date.now()
            await pt.step()
            const tb = Date.now()
            console.log(tb - ta)
        }

        await display.draw()


        window.requestAnimationFrame(frame)
    }

    frame()
}