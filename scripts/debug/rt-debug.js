
function initDebug(device, canvas, scene) {
    const CANVAS = initCanvas(device, canvas)

    let r1   = 3.1415 * 1.8
    let r2   = .6
    let dist = 10

    let lookAt = [0, 0, 0]
    let position = [ 10, 10, 10
        //lookAt[0] + Math.cos(r1) * Math.cos(r2) * dist,
        //lookAt[1] + Math.sin(r1) * Math.cos(r2) * dist,
        //lookAt[2] + Math.sin(r2) * dist
    ]

    const { VS, FS, CS } = SRC()

    const IMG_BUFFER = device.createBuffer({
        size: CANVAS.w * CANVAS.h * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    })

    const BG_LAYOUT = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                buffer: {
                    type: "storage"
                }
            }
        ]
    })

    const BG = device.createBindGroup({
        layout: BG_LAYOUT,
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                resource: {
                    buffer: IMG_BUFFER
                }
            }
        ]
    })

    const DRAW_SM = device.createShaderModule({
        code: VS + FS
    })

    const DRAW_PIPELINE = device.createRenderPipeline({
        layout: device.createPipelineLayout({bindGroupLayouts: [BG_LAYOUT]}),
        vertex: {
            module: DRAW_SM,
            entryPoint: "vs"
        },
        fragment: {
            module: DRAW_SM,
            entryPoint: "fs",
            targets: [
                {
                    format: CANVAS.presentationFormat
                }
            ]
        }
    })

    const COMPUTE_SM = device.createShaderModule({
        code: CS + scene.kernels.getNearestHitCode(1)
    })

    const COMPUTE_PIPELINE = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [BG_LAYOUT, scene.kernels.getSceneBindGroupInfo().bindGroupLayout]
        }),
        compute: {
            module: COMPUTE_SM,
            entryPoint: "main"
        }
    })

    return { draw }

    async function draw() {
        {
            const CE = device.createCommandEncoder()
            const P  = CE.beginComputePass()
            P.setPipeline(COMPUTE_PIPELINE)
            P.setBindGroup(0, BG)
            P.setBindGroup(1, scene.kernels.getSceneBindGroupInfo().bindGroup)
            P.dispatchWorkgroups(Math.ceil(CANVAS.w / 8), Math.ceil(CANVAS.h / 8))
            P.end()
            device.queue.submit([CE.finish()])
        }
        {
            const CE = device.createCommandEncoder()
            const  P = CE.beginRenderPass({
                colorAttachments: [
                    {
                        view: CANVAS.ctx.getCurrentTexture().createView(),
                        clearValue: {r: 1., g: 0., b: 0., a: 1.},
                        loadOp: "clear", 
                        storeOp: "store"
                    }
                ]
            })
            P.setPipeline(DRAW_PIPELINE)
            P.setBindGroup(0, BG)
            P.draw(6)
            P.end()
            device.queue.submit([CE.finish()])
        }

        await device.queue.onSubmittedWorkDone()

        return
    }

    function SRC() {
        let CS = /* wgsl */ `
        @group(0) @binding(0) var<storage, read_write> image_buffer : array<vec4f>;

        const Pi      = 3.14159265358979323846;
        const InvPi   = 0.31830988618379067154;
        const Inv2Pi  = 0.15915494309189533577;
        const Inv4Pi  = 0.07957747154594766788;
        const PiOver2 = 1.57079632679489661923;
        const PiOver4 = 0.78539816339744830961;
        const Sqrt2   = 1.41421356237309504880;

        const sw_f : vec2f = vec2f(${CANVAS.w}., ${CANVAS.h}.);
        const sw_u : vec2u = vec2u(${CANVAS.w}u, ${CANVAS.h}u);

        const     fov :   f32 = 80.f;
        const  sinfov :   f32 = sin(.5 * fov * Pi / 180.f);
        const  aspect :   f32 = ${CANVAS.w / CANVAS.h}f;

        const  lookAt : vec3f = vec3f(${lookAt[0]},${lookAt[1]}, ${lookAt[2]});
        const     pos : vec3f = vec3f(${position[0]},${position[1]},${position[2]});
        const forward : vec3f = normalize(lookAt - pos);
        const   right : vec3f = normalize(vec3f(forward.y, -forward.x, 0.));
        const      up : vec3f = cross(right, forward);

        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) global_id : vec3u) {
            if (any(global_id.xy >= sw_u)) {return;}

            var coord : vec2i = vec2i(global_id.xy);

            var img_idx : i32 = dot(coord, vec2i(1, ${CANVAS.w}));

            var o : vec3f = vec3f(-5., 0., 0.);
            var d : vec3f = vec3f( 1., 0., 0.);

            getCameraRay(vec2f(coord), &o, &d);

            var res : BVHHitResult = intersect_bvh(o, d);

            var col : vec3f = vec3f(0.f);

            if (res.hit_dis > 1e5f) {
                col = vec3f(0.f);
            } else {
                col = vec3f(f32(res.hit_obj) / 20.f, f32(res.hit_tri) / 5000.f, 0.f);
            }

            image_buffer[img_idx] = vec4f(
                vec3f(col), 1.
            );
        }
        
        fn getCameraRay(coord : vec2f, o : ptr<function, vec3f>, d : ptr<function, vec3f>) {
            var sspace : vec2f = coord / sw_f; sspace = sspace * 2. - vec2f(1.); sspace.y *= -1.;
            var local  : vec3f = vec3f(
                aspect * sspace.x * sinfov,
                1.,
                sspace.y * sinfov
            );
            

            *o = pos;
            *d = toWorld(right, forward, up, normalize(local));
        }
        
        fn toWorld(v_x : vec3f, v_y : vec3f, v_z : vec3f, w : vec3f) -> vec3f {
            return v_x * w.x + v_y * w.y + v_z * w.z;
        }`

        let VS = /* wgsl */ `
        @vertex
        fn vs(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4f {
            switch(vertexIndex) {
                case 0u: {
                    return vec4f(1., 1., 0., 1.);}
                case 1u: {
                    return vec4f(-1., 1., 0., 1.);}
                case 2u: {
                    return vec4f(-1., -1., 0., 1.);}
                case 3u: {
                    return vec4f(1., -1., 0., 1.);}
                case 4u: {
                    return vec4f(1., 1., 0., 1.);}
                case 5u: {
                    return vec4f(-1., -1., 0., 1.);}
                default: {
                    return vec4f(0., 0., 0., 0.);}
            }
        }`

        let FS = /* wgsl */ `
        @group(0) @binding(0) var<storage, read_write> image_buffer : array<vec4f>;

        fn lum(z : vec3f) -> f32 {
            return dot(z, vec3f(.2126, .7152, .0722));
        }

        @fragment
        fn fs(@builtin(position) fragCoord : vec4f) -> @location(0) vec4f {
            var coord : vec2i = vec2i(fragCoord.xy);
            var img_idx : i32 = dot(coord, vec2i(1, ${CANVAS.w}));
            return vec4f(image_buffer[img_idx]);
        }`

        return { CS, VS, FS }
    }



    function initCanvas(device, canvas) {
        let ctx = canvas.getContext("webgpu")
    
        let presentationFormat = navigator.gpu.getPreferredCanvasFormat()
        ctx.configure({device, format: presentationFormat})
    
        const w = Math.ceil(canvas.clientWidth  * 1.5) 
        const h = Math.ceil(canvas.clientHeight * 1.5) 
    
        canvas.width  = w
        canvas.height = h
    
        return {
            ctx, presentationFormat, w, h
        }
    }
}