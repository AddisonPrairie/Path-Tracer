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
    scene.registerMesh({ file: floorFile })
    
    for (var i = 0; i < 100; i++) {
        scene.instanceMesh(2, 
            [(i % 10) * 2, (Math.floor(i / 10)) * 2, 0],
            [Math.PI / 2, 0, Math.random() * Math.PI * 2.],
            [1, 1, 1], 1
        )
    }

    scene.instanceMesh(3, [10, 10, -1.3], [0, 0, 0], [5, 5, 5], 0)

    await scene.build()

    const pt = initPathTracer({ 
        device, scene,
        image: {
            width: 2048, height: 2048
        },
        camera: {
            lookAt: [10, 10, 0],
            position: [15, 15, 5],
            fov: 60
        },
        settings: {
            samples: 128
        }
    })

    const display = initDisplay({
        device, 
        canvas: document.querySelector("#canvas"),
        image: {
            buffer: pt.getImageBuffer(),
            width: 2048, height: 2048
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