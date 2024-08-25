

window.onload = async () => {
    // initialize WebGPU
    const adapter = await navigator.gpu?.requestAdapter()
    const device  = await adapter?.requestDevice()

    if (!device) {
        alert("browser does not support WebGPU!")
        return
    }

    let cowFile = await fetch("media/cow.obj").then(f => f.text())

    const scene = initScene(device)

    scene.registerMesh({file: cowFile})
    scene.registerMesh({file: cowFile})
    scene.registerMesh({file: cowFile})

    scene.instanceMesh(0, [0, 0, 0], [0, 0, 0], [1, 1, 1])
    scene.instanceMesh(2, [0, 0, 0], [0, 0, 0], [1, 1, 1])

    await scene.build()
}


async function readBackBuffer(device, buffer) {
    const readBuffer = device.createBuffer({
        size: buffer.size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    })
    const CE = device.createCommandEncoder()
    CE.copyBufferToBuffer(buffer, 0, readBuffer, 0, buffer.size)
    device.queue.submit([CE.finish()])
    await readBuffer.mapAsync(GPUMapMode.READ)
    const ret = new ArrayBuffer(buffer.size)
    new Int32Array(ret).set(new Int32Array(readBuffer.getMappedRange()))
    readBuffer.destroy()
    return ret
}