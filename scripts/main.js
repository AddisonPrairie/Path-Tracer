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
    let buddhaFile = await fetch("media/buddha.obj").then(f => f.text())

    const scene = initScene(device)

    const t0 = Date.now()

    const cubeModel = scene.registerMesh({ file: cubeFile })
    const floorModel = scene.registerMesh({ file: floorFile })
    const buddhaModel = scene.registerMesh({ file: buddhaFile })

    const whiteDiffuse = scene.addMaterial("lambert_diffuse", {color: {r: .8, g: .8, b: .8}})
    const redDiffuse   = scene.addMaterial("lambert_diffuse", {color: {r: .5, g: 0., b: 0.}})
    const greenDiffuse = scene.addMaterial("lambert_diffuse", {color: {r: 0., g: .5, b: 0.}})
    const ggxSmith     = scene.addMaterial("ggx_smith", {color: {r: .5, g: .5, b: .5}, roughness: .05})
    const mirror = scene.addMaterial("mirror", {color: {r: .6, g: .6, b: .6}})
    
    //scene.instanceMesh(buddhaModel, [0, 0, 1.5], [Math.PI / 2, 0, 0], [5, 5, 5], whiteDiffuse)
    scene.instanceMesh(cubeModel, [-3, 3, 1.2], [0, 0, Math.PI / 4], [1.2, 1.2, 1.2], whiteDiffuse)
    scene.instanceMesh(cubeModel, [3, 1, 2], [0, 0, Math.PI / 3], [1, 1, 2], whiteDiffuse)

    scene.instanceMesh(floorModel, [ 0,  0,  0], [0, 0, 0], [5, 5, 5], whiteDiffuse)
    scene.instanceMesh(floorModel, [-5,  0,  5], [0, Math.PI / 2, 0], [5, 5, 5], greenDiffuse)
    scene.instanceMesh(floorModel, [5,  0,  5], [0, Math.PI / 2, 0], [5, 5, 5], redDiffuse)
    scene.instanceMesh(floorModel, [ 0,  0, 10], [0, 0, 0], [5, 5, 5], whiteDiffuse)
    scene.instanceMesh(floorModel, [ 0, -5,  5], [Math.PI / 2, 0, 0], [5, 5, 5], ggxSmith)

    scene.addLight("rectangle", {
        position: {x: 0, y: 0, z: 3},
        target: {x: 10, y: 0, z: 0},
        scale: {x: 1.5, y: 1.5},
        le: {r: 25, g: 25, b: 15}
    })

    await scene.build()

    const {w, h} = {w:512, h:512}

    const pt = initPathTracer({ 
        device, scene,
        image: {
            width: w, height: h
        },
        camera: {
            lookAt: [0, 0, 5],
            position: [0, 13, 5],
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