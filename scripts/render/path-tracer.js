function initPathTracer(params) {
    const device = params.device

    const NUM_PATHS = 1_000_000
    const BYTES_PER_PATH = 108


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

    return { step }

    function step() {
        logicKernel.execute()
        cameraKernel.execute()
        materialKernel.execute()
        rayTraceKernel.execute()
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

            material_queue : array<i32, ${NUM_PATHS}>,
            camera_queue   : array<i32, ${NUM_PATHS}>,

            ray_trace_queue : array<i32, ${NUM_PATHS}>,
        };

        struct QueuesStage2 {
            stage_2_queue_size : array<i32, 2>,
            stage_3_queue_size : array<atomic<i32>, 1>,
            f_0 : i32,

            material_queue : array<i32, ${NUM_PATHS}>,
            camera_queue   : array<i32, ${NUM_PATHS}>,

            ray_trace_queue : array<i32, ${NUM_PATHS}>,
        };

        struct QueuesStage3 {
            stage_2_queue_size : array<i32, 2>,
            stage_3_queue_size : array<i32, 1>,
            f_0 : i32,

            material_queue : array<i32, ${NUM_PATHS}>,
            camera_queue   : array<i32, ${NUM_PATHS}>,

            ray_trace_queue : array<i32, ${NUM_PATHS}>,
        }
        `
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
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
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
            
            
            /*const stage2QueueCountBuffer = device.createBuffer({
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
            })*/
        }
    }
}