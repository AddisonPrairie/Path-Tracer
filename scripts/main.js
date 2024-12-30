window.onload = async () => {
    // initialize WebGPU
    const adapter = await navigator.gpu?.requestAdapter()
    const device  = await adapter?.requestDevice()

    if (!device) {
        alert("browser does not support WebGPU!")
        return
    }

    let floorFile  = await fetch("media/floor.obj").then(f => f.text())
    let buddhaFile = await fetch("media/buddha.obj").then(f => f.text())

    const scene = initScene(device)

    const t0 = Date.now()

    scene.registerMesh({ file: buddhaFile })
    scene.registerMesh({ file: floorFile })
    
    scene.instanceMesh(0, [0, 0, 2.9], [Math.PI / 2, 0, 0], [7, 7, 7], 0)

    scene.instanceMesh(1, [ 0,  0,  0], [0, 0, 0], [.5, .5, .5], 0)
    scene.instanceMesh(1, [ 0,  0, 10], [0, 0, 0], [.5, .5, .5], 0)
    scene.instanceMesh(1, [ 0, -5,  5], [Math.PI / 2, 0, 0], [.5, .5, .5], 0)
    scene.instanceMesh(1, [-5,  0,  5], [0, Math.PI / 2, 0], [.5, .5, .5], 0)
    scene.instanceMesh(1, [ 5,  0,  5], [0, Math.PI / 2, 0], [.5, .5, .5], 0)


    await scene.build()

    console.log(scene)

    const pt = initPathTracer({ 
        device, scene,
        image: {
            width: 512, height: 512
        },
        camera: {
            lookAt: [0, 0, 5],
            position: [0, 12, 5],
            fov: 60
        },
        settings: {
            samples: 1024
        }
    })

    const display = initDisplay({
        device, 
        canvas: document.querySelector("#canvas"),
        image: {
            buffer: pt.getImageBuffer(),
            width: 512, height: 512
        }
    })

    async function frame() {
        for (var i = 0; i < 1; i++) {
            const ta = Date.now()
            await pt.step()
            const tb = Date.now()
            //console.log(tb - ta)
        }

        await display.draw()

        window.requestAnimationFrame(frame)
    }

    frame()
}