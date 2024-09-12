function initPathTracer(params) {
    const device = params.device

    const NUM_PATHS = params.image.width * params.image.height//1_000_000
    const BYTES_PER_PATH = 112

    const DEBUG_MODE = true


    const bindGroups = {}
    const bindGroupLayouts = {}
    const buffers = {}

    createBindGroups()

    const logicKernel = initLogicKernel({
        bindGroups, bindGroupLayouts, device,
        numPaths: NUM_PATHS,
        sharedStructCode: SHARED_STRUCTS_CODE()
    })

    const cameraKernel = initCameraKernel({
        bindGroups, bindGroupLayouts, device,
        numPaths: NUM_PATHS,
        sharedStructCode: SHARED_STRUCTS_CODE()
    })

    const materialKernel = initMaterialKernel({
        bindGroups, bindGroupLayouts, device,
        scene: params.scene,
        numPaths: NUM_PATHS,
        sharedStructCode: SHARED_STRUCTS_CODE()
    })

    const rayTraceKernel = initRayTraceKernel({
        bindGroups, bindGroupLayouts, device,
        scene: params.scene,
        numPaths: NUM_PATHS,
        sharedStructCode: SHARED_STRUCTS_CODE()
    })

    let renderInfo = { numSteps: 0 }

    return { step, getImageBuffer }

    async function step() {

        if (renderInfo.numSteps < 2) { 
            // write to the uniform buffer

            const uniformBuffer = new ArrayBuffer(64)
            const dataView = new DataView(uniformBuffer)

            dataView.setInt32( 0, params.image.width,  true)
            dataView.setInt32( 4, params.image.height, true)

            dataView.setInt32( 8, renderInfo.numSteps == 0 ? 1 : 0, true)

            dataView.setFloat32(12, params.camera.fov, true)

            dataView.setFloat32(16, params.camera.position[0], true)
            dataView.setFloat32(20, params.camera.position[1], true)
            dataView.setFloat32(24, params.camera.position[2], true)

            dataView.setFloat32(32, params.camera.lookAt[0], true)
            dataView.setFloat32(36, params.camera.lookAt[1], true)
            dataView.setFloat32(40, params.camera.lookAt[2], true)

            device.queue.writeBuffer(buffers.uniforms, 0, uniformBuffer, 0)
        }

        { // clear out the queue counts from last step
            if (renderInfo.numSteps > 0) {
                device.queue.writeBuffer(buffers.queues, 0, new Int32Array([0, 0, 0]), 0)
            }
        }
        console.log("-------")
        {
            const ta = Date.now()
            await logicKernel.execute()
            const tb = Date.now()

            console.log("logic ", tb - ta)
        }
        {
            const ta = Date.now()
            await cameraKernel.execute()
            const tb = Date.now()

            console.log("camera ", tb - ta)
        }
        {
            const ta = Date.now()
            await materialKernel.execute()
            const tb = Date.now()

            console.log("material ", tb - ta)
        }
        {
            const ta = Date.now()
            await rayTraceKernel.execute()
            const tb = Date.now()

            console.log("ray trace ", tb - ta)
        }
        console.log("-------")

        renderInfo.numSteps++
    }

    function getImageBuffer() {
        return buffers.image
    }

    function SHARED_STRUCTS_CODE() {
        return /* wgsl */ `
        struct PathState {
            pixel_index : array<i32, ${NUM_PATHS}>, // 4 bytes
            num_bounces : array<i32, ${NUM_PATHS}>, // 4 bytes

            random_seed : array<f32, ${NUM_PATHS}>, // 4 bytes

            path_throughput : array<vec3f, ${NUM_PATHS}>, // 16 bytes

            material_throughput_pdf : array<vec4f, ${NUM_PATHS}>, // 16 bytes

            path_o : array<vec3f, ${NUM_PATHS}>, // 16 bytes
            path_d : array<vec3f, ${NUM_PATHS}>, // 16 bytes

            hit_obj : array<i32, ${NUM_PATHS}>, // 16 bytes
            hit_tri : array<i32, ${NUM_PATHS}>, // 16 bytes

            flags : array<u32, ${NUM_PATHS}>, // 4 bytes

            // Description of flags:

            /* 
             * 1 <<  0 : material evaluation
             * 1 <<  1 : 
             */
        };
        
        struct Uniforms {
            image_size : vec2i, // > 8 bytes

            // != 0 iff this is the first time everything is executed
            first_sample : i32, // > 12 bytes
            
            camera_fov : f32, // > 16 bytes
            camera_position : vec3f, // > 32 bytes
            camera_look_at  : vec3f, // > 48 bytes
        };
        
        struct QueuesStage1 {
            stage_2_queue_size : array<atomic<i32>, 2>,
            stage_3_queue_size : array<atomic<i32>, 1>,
            f_0 : i32,

            camera_queue   : array<i32, ${NUM_PATHS}>,
            material_queue : array<i32, ${NUM_PATHS}>,

            ray_trace_queue : array<i32, ${NUM_PATHS}>,
        };

        struct QueuesStage2 {
            stage_2_queue_size : array<i32, 2>,
            stage_3_queue_size : array<atomic<i32>, 1>,
            f_0 : i32,

            camera_queue   : array<i32, ${NUM_PATHS}>,
            material_queue : array<i32, ${NUM_PATHS}>,

            ray_trace_queue : array<i32, ${NUM_PATHS}>,
        };

        struct QueuesStage3 {
            stage_2_queue_size : array<i32, 2>,
            stage_3_queue_size : array<i32, 1>,
            f_0 : i32,

            camera_queue   : array<i32, ${NUM_PATHS}>,
            material_queue : array<i32, ${NUM_PATHS}>,

            ray_trace_queue : array<i32, ${NUM_PATHS}>,
        }`
    }

    function createBindGroups() {
        {// create bind group info for scene
            const info = params.scene.kernels.getSceneBindGroupInfo()
            bindGroupLayouts.scene = info.bindGroupLayout
            bindGroups.scene = info.bindGroup
        }

        {// create bind group info for image
            bindGroupLayouts.image = device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "storage"
                        }
                    }
                ]
            })

            buffers.image = device.createBuffer({
                size: params.image.width * params.image.height * 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            })

            bindGroups.image = device.createBindGroup({
                layout: bindGroupLayouts.image,
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: buffers.image
                        }
                    }
                ]
            })
        }
    
        {// create bind group info for path state
            bindGroupLayouts.pathState = device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "storage"
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "uniform"
                        }
                    }
                ]
            })

            buffers.uniforms = device.createBuffer({
                size: 64,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            })

            buffers.pathState = device.createBuffer({
                size: BYTES_PER_PATH * NUM_PATHS,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | (DEBUG_MODE ? GPUBufferUsage.COPY_SRC : 0)
            })

            bindGroups.pathState = device.createBindGroup({
                layout: bindGroupLayouts.pathState,
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: buffers.pathState
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: buffers.uniforms
                        }
                    }
                ]
            })
        }
    
        {// create bind group info for queues
            buffers.queues = device.createBuffer({
                size: 16 + NUM_PATHS * 12,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | (DEBUG_MODE ? GPUBufferUsage.COPY_SRC : 0)
            })

            bindGroupLayouts.queues = device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "storage"
                        }
                    }
                ]
            })

            bindGroups.queues = device.createBindGroup({
                layout: bindGroupLayouts.queues,
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: buffers.queues
                        }
                    }
                ]
            })
        }
    }
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