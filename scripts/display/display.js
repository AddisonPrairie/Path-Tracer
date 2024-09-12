// draws a buffer to a canvas
function initDisplay(params) {
    const device = params.device

    const canvas = initCanvas()

    const BG_LAYOUT = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
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
                visibility: GPUShaderStage.FRAGMENT,
                resource: {
                    buffer: params.image.buffer
                }
            }
        ]
    })

    const SM = device.createShaderModule({
        code: SRC()
    })

    const PIPELINE = device.createRenderPipeline({
        layout: device.createPipelineLayout({bindGroupLayouts: [BG_LAYOUT]}),
        vertex: {
            module: SM,
            entryPoint: "vs"
        },
        fragment: {
            module: SM,
            entryPoint: "fs",
            targets: [
                {
                    format: canvas.presentationFormat
                }
            ]
        }
    })

    return { draw }

    function draw() {
        const CE = device.createCommandEncoder()
        const  P = CE.beginRenderPass({
            colorAttachments: [
                {
                    view: canvas.ctx.getCurrentTexture().createView(),
                    clearValue: {r: 1., g: 0., b: 0., a: 1.},
                    loadOp: "clear", 
                    storeOp: "store"
                }
            ]
        })
        P.setPipeline(PIPELINE)
        P.setBindGroup(0, BG)
        P.draw(6)
        P.end()
        device.queue.submit([CE.finish()])
    }

    function SRC() {
        return /* wgsl */ `
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
        }

        @group(0) @binding(0) var<storage, read_write> image_buffer : array<vec4f>;

        fn lum(z : vec3f) -> f32 {
            return dot(z, vec3f(.2126, .7152, .0722));
        }

        @fragment
        fn fs(@builtin(position) fragCoord : vec4f) -> @location(0) vec4f {
            var coord : vec2i = vec2i(fragCoord.xy);
            var img_idx : i32 = dot(coord, vec2i(1, ${params.image.width}));

            var pixel : vec4f = image_buffer[img_idx];

            var col : vec3f = pixel.xyz / pixel.w;

            col = col / (1.f + lum(col));
            return vec4f(pow(col, vec3f(1. / 2.2)), 1.);
        }`
    }

    function initCanvas() {
        let ctx = params.canvas.getContext("webgpu")
    
        let presentationFormat = navigator.gpu.getPreferredCanvasFormat()
        ctx.configure({device , format: presentationFormat})
    
        const width = Math.ceil(params.image.width) 
        const height = Math.ceil(params.image.height) 
    
        params.canvas.width  = width
        params.canvas.height = height
    
        return {
            ctx, presentationFormat
        }
    }
}