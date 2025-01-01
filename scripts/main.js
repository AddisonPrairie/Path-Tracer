window.onload = async () => {
    // initialize WebGPU
    const adapter = await navigator.gpu?.requestAdapter()
    const device  = await adapter?.requestDevice()

    if (!device) {
        alert("browser does not support WebGPU!")
        return
    }

    let floorFile  = await fetch("media/square.obj").then(f => f.text())
    let cubeFile = await fetch("media/cube.obj").then(f => f.text())

    const scene = initScene(device)

    const t0 = Date.now()

    //scene.registerMesh({ file: buddhaFile })
    const cubeModel = scene.registerMesh({ file: cubeFile })
    const floorModel = scene.registerMesh({ file: floorFile })

    const whiteDiffuse = scene.addMaterial("lambert_diffuse", {color: {r: .8, g: .8, b: .8}})
    const redDiffuse   = scene.addMaterial("lambert_diffuse", {color: {r: .5, g: 0., b: 0.}})

    console.log(whiteDiffuse)
    console.log(redDiffuse)
    
    scene.instanceMesh(cubeModel, [0, 0, 1.5], [0, 0, Math.PI / 4.], [1.5, 1.5, 1.5], whiteDiffuse)
    scene.instanceMesh(cubeModel, [-3, 3, 1.2], [0, 0, 0], [1.2, 1.2, 1.2], whiteDiffuse)

    scene.instanceMesh(floorModel, [ 0,  0,  0], [0, 0, 0], [5, 5, 5], whiteDiffuse)
    scene.instanceMesh(floorModel, [ 0,  0, 10], [0, 0, 0], [5, 5, 5], redDiffuse)
    scene.instanceMesh(floorModel, [ 0, -5,  5], [Math.PI / 2, 0, 0], [5, 5, 5], whiteDiffuse)

    //scene.instanceMesh(floorModel, [-5,  0,  5], [0, Math.PI / 2, 0], [5, 5, 5], 3)
    //scene.instanceMesh(floorModel, [ 5,  0,  5], [0, Math.PI / 2, 0], [5, 5, 5], 3)
    //scene.instanceMesh(floorModel, [3, 0, 9.9999], [0, 0, 0], [1, 1, 1], 1)
    //scene.instanceMesh(floorModel, [-3, 0, 9.9999], [0, 0, 0], [1, 1, 1], 1)
    /*
    scene.addLight("rectangle", {
        position: {x: 0, y: 0, z: 3.5},
        target: {x: 0, y: 0, z: 0},
        scale: {x: 1, y: 1}
    })
    */
    //scene.instanceMesh(1, [0, 0, 9.999], [0, 0, 0], [.2, .2, .2], 1)
    //scene.instanceMesh(0, [2, 0, 3], [0, 0, Math.PI / 4.], [1.5, 1.5, 3], 0)
    //scene.instanceMesh(0, [])

    await scene.build()

    const {w, h} = {w:512, h:512}

    const pt = initPathTracer({ 
        device, scene,
        image: {
            width: w, height: h
        },
        camera: {
            lookAt: [0, 0, 5],
            position: [8, 8, 7],
            fov: 45
        },
        settings: {
            samples: 4096
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