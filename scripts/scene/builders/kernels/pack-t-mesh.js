
function initPackTMeshKernel(device) {
    // shader parameters
    const WG_SIZE = 64

    // create bind group layout, shader module and pipeline
    const I_BG_LAYOUT = device.createBindGroupLayout({
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
        ]
    })

    const O_BG_LAYOUT = device.createBindGroupLayout({
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
                    type: "uniform"
                }
            }
        ]
    })

    const SM = device.createShaderModule({
        code: SRC(),
        label: "Pack Triangle Mesh shader module"
    })

    const PIPELINE = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [I_BG_LAYOUT, O_BG_LAYOUT]
        }),
        compute: {
            module: SM,
            entryPoint: "pack_triangle_mesh"
        }
    })

    return { execute }

    async function execute(bvhs) {
        const NUM_MESHES = bvhs.length

        // compute mesh offsets and write to GPU
        const OFFSET_BUFFER = device.createBuffer({
            size: NUM_MESHES * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        })

        let offsets = []
        let  oStart = 0
        for (var i = 0; i < NUM_MESHES; i++) {
            offsets.push(oStart)
            oStart += bvhs[i].numTriangles
        }

        device.queue.writeBuffer(OFFSET_BUFFER, 0, new Int32Array(offsets))

        // create other buffers used in execution
        const PACKED_BVH_BUFFER = device.createBuffer({
            size: oStart * 64,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        })

        const PACKED_TRI_BUFFER = device.createBuffer({
            size: oStart * 48,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        })

        const UNIFORM_BUFFER = device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        })

        const O_BG = device.createBindGroup({
            layout: O_BG_LAYOUT,
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    resource: {
                        buffer: PACKED_TRI_BUFFER
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    resource: {
                        buffer: PACKED_BVH_BUFFER
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    resource: {
                        buffer: OFFSET_BUFFER
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    resource: {
                        buffer: UNIFORM_BUFFER
                    }
                }
            ]
        })

        for (var i = 0; i < NUM_MESHES; i++) {
            device.queue.writeBuffer(UNIFORM_BUFFER, 0, new Int32Array([i, bvhs[i].numTriangles]))

            const I_BG = device.createBindGroup({
                layout: I_BG_LAYOUT,
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: bvhs[i].triangleBuffer
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: bvhs[i].bvhBuffer
                        }
                    }
                ]
            })

            const CE = device.createCommandEncoder()
            const  P = CE.beginComputePass()

            P.setPipeline(PIPELINE)
            P.setBindGroup(0, I_BG)
            P.setBindGroup(1, O_BG)
            P.dispatchWorkgroups(Math.ceil(bvhs[i].numTriangles / WG_SIZE))
            P.end()

            device.queue.submit([CE.finish()])

            await device.queue.onSubmittedWorkDone()
        }

        return { PACKED_BVH_BUFFER, PACKED_TRI_BUFFER }
    }

    function SRC() {
        return /* wgsl */ `

        struct Triangle {
            v0 : vec3f,
            v1 : vec3f,
            v2 : vec3f
        };

        struct BVHNode {
            aabb_l_min : vec3f,
               l_child :   i32,
            aabb_l_max : vec3f,
                   f_1 :   i32,
            aabb_r_min : vec3f,
               r_child :   i32,
            aabb_r_max : vec3f,
                   f_2 :   i32
        };

        struct Uniforms {
            mesh_idx : i32,
            num_tris : i32
        };

        @group(0) @binding(0) var<storage, read_write> src_triangles : array<Triangle>;
        @group(0) @binding(1) var<storage, read_write> src_bvh : array<BVHNode>;

        @group(1) @binding(0) var<storage, read_write> packed_triangles : array<Triangle>;
        @group(1) @binding(1) var<storage, read_write> packed_bvh : array<BVHNode>;
        @group(1) @binding(2) var<storage, read_write> offsets : array<i32>;
        @group(1) @binding(3) var<uniform> uniforms : Uniforms;


        @compute @workgroup_size(${WG_SIZE})
        fn pack_triangle_mesh(@builtin(global_invocation_id) global_id : vec3u) {
            var idx : i32 = i32(global_id.x);
            if (idx >= uniforms.num_tris) {
                return;
            }

            var offset : i32 = offsets[uniforms.mesh_idx];

            // move one triangle and one bvh node
            packed_triangles[offset + idx] = src_triangles[idx];

            var node : BVHNode = src_bvh[idx];

            if (node.l_child < 0) {
                node.l_child -= offset;
            } else {
                node.l_child += offset;
            }

            if (node.r_child < 0) {
                node.r_child -= offset;
            } else {
                node.r_child += offset;
            }

            packed_bvh[offset + idx] = node;
        }`
    }
}