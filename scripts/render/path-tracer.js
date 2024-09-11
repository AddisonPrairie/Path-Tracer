function initPathTracer(params) {
    const device = params.device

    const NUM_PATHS = 1_000_000
    const BYTES_PER_PATH = 4


    const bindGroups = {}
    const bindGroupLayouts = {}
    const buffers = {}

    createBindGroups()

    const rayTraceKernel = initRayTraceKernel({
        bindGroups, bindGroupLayouts, device,
        scene: params.scene,
        numPaths: NUM_PATHS,
        sharedStructCode: SHARED_STRUCTS_CODE()
    })

    const cameraKernel = initCameraKernel({
        bindGroups, bindGroupLayouts, device,
        numPaths: NUM_PATHS,
        sharedStructCode: SHARED_STRUCTS_CODE()
    })

    function SHARED_STRUCTS_CODE() {
        return /* wgsl */ `
        struct PathState {
            pixel_index : array<i32, ${params.numActivePaths}>,
            num_bounces : array<i32, ${params.numActivePaths}>,

            random_seed : array<f32, ${params.numActivePaths}>,

            path_throughput : array<vec3f, ${params.numActivePaths}>,

            material_throughput_pdf : array<vec4f, ${params.numActivePaths}>,

            path_o : array<vec3f, ${params.numActivePaths}>,
            path_d : array<vec3f, ${params.numActivePaths}>,

            hit_obj : array<i32, ${params.numActivePaths}>,
            hit_tri : array<i32, ${params.numActivePaths}>
        };
        
        struct Uniforms {
            image_size : vec2i,

            camera_position : vec3f,
            camera_look_at  : vec3f,
        };`
    }

    function createBindGroups() {
        {// create bind group info for scene
            const info = params.scene.kernels.getSceneBindGroupInfo()
            bindGroupLayouts.scene = info.bindGroupLayout
            bindGroups.scene = info.bindGroup
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
                    }
                ]
            })
            bindGroups.pathState = device.createBindGroup({
                layout: bindGroupLayouts.pathState,
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: device.createBuffer({
                                size: BYTES_PER_PATH * NUM_PATHS,
                                usage: GPUBufferUsage.STORAGE
                            })
                        }
                    }
                ]
            })
        }
    
        {// create bind group info for queues
            const stage2QueueCountBuffer = device.createBuffer({
                size: 8,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            })
    
            const stage3QueueCountBuffer = device.createBuffer({
                size: 8,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            })
    
            const cameraQueueBuffer = device.createBuffer({
                size: 4 * NUM_PATHS,
                usage: GPUBufferUsage.STORAGE
            })
    
            const materialQueueBuffer = device.createBuffer({
                size: 4 * NUM_PATHS,
                usage: GPUBufferUsage.STORAGE
            })
    
            const rayTraceQueueBuffer = device.createBuffer({
                size: 4 * NUM_PATHS,
                usage: GPUBufferUsage.STORAGE
            })
    
            // also store the two queue count buffers so they can be cleared later
            buffers.stage2QueueCountBuffer = stage2QueueCountBuffer
            buffers.stage3QueueCountBuffer = stage3QueueCountBuffer
    
            bindGroupLayouts.stage2Queues = device.createBindGroupLayout({
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
                            type: "storage"
                        }
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "storage"
                        }
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "storage"
                        }
                    },
                    {
                        binding: 4,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "storage"
                        }
                    }
                ]
            })
    
            bindGroups.stage2Queues = device.createBindGroup({
                layout: bindGroupLayouts.stage2Queues,
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: stage2QueueCountBuffer
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: cameraQueueBuffer
                        }
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: materialQueueBuffer
                        }
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: stage3QueueCountBuffer
                        }
                    },
                    {
                        binding: 4,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: rayTraceQueueBuffer
                        }
                    }
                ]
            })
    
            bindGroupLayouts.stage3Queues = device.createBindGroupLayout({
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
                            type: "storage"
                        }
                    }
                ]
            })
    
            bindGroups.stage3Queues = device.createBindGroup({
                layout: bindGroupLayouts.stage3Queues,
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: stage3QueueCountBuffer
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: rayTraceQueueBuffer
                        }
                    }
                ]
            })
        }
    }
}