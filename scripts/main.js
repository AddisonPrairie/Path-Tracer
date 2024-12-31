window.onload = async () => {
    // initialize WebGPU
    const adapter = await navigator.gpu?.requestAdapter()
    const device  = await adapter?.requestDevice()

    if (!device) {
        alert("browser does not support WebGPU!")
        return
    }

    let floorFile  = await fetch("media/floor.obj").then(f => f.text())
    let cubeFile = await fetch("media/cube.obj").then(f => f.text())

    const scene = initScene(device)

    const t0 = Date.now()

    //scene.registerMesh({ file: buddhaFile })
    scene.registerMesh({ file: cubeFile })
    scene.registerMesh({ file: floorFile })
    
    //scene.instanceMesh(0, [0, 0, 2.9], [Math.PI / 2, 0, 0], [7, 7, 7], 0)

    scene.instanceMesh(1, [ 0,  0,  0], [0, 0, 0], [.5, .5, .5], 0)
    scene.instanceMesh(1, [ 0,  0, 10], [0, 0, 0], [.5, .5, .5], 0)
    scene.instanceMesh(1, [ 0, -5,  5], [Math.PI / 2, 0, 0], [.5, .5, .5], 0)
    scene.instanceMesh(1, [-5,  0,  5], [0, Math.PI / 2, 0], [.5, .5, .5], 3)
    scene.instanceMesh(1, [ 5,  0,  5], [0, Math.PI / 2, 0], [.5, .5, .5], 2)

    scene.instanceMesh(1, [0, 0, 9.999], [0, 0, 0], [.2, .2, .2], 1)

    scene.instanceMesh(0, [2, 0, 3], [0, 0, Math.PI / 4.], [1.5, 1.5, 3], 0)

    await scene.build()

    const {w, h} = {w:512, h:512}

    const pt = initPathTracer({ 
        device, scene,
        image: {
            width: w, height: h
        },
        camera: {
            lookAt: [0, 0, 5],
            position: [0, 15, 5],
            fov: 45
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
            width: w, height: h
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