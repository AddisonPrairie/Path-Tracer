/*  
    4 x 4 matrices are laid out as
    [
        <=== collumn 1 ===>,
        <=== collumn 2 ===>,
        <=== collumn 3 ===>,
        <=== collumn 4 ===>
    ]
    To work with WebGPU/WGSL
*/
function getTransformedBoundingBox(transformMatrices, box) {
    let points = [
        [box.min[0], box.min[1], box.min[2]],
        [box.max[0], box.min[1], box.min[2]],
        [box.min[0], box.max[1], box.min[2]],
        [box.min[0], box.min[1], box.max[2]],
        [box.max[0], box.min[1], box.max[2]],
        [box.max[0], box.max[1], box.min[2]],
        [box.min[0], box.max[1], box.max[2]],
        [box.max[0], box.max[1], box.max[2]],
    ]

    function transformPoint(pnt) {
        let l2w = transformMatrices.localToWorld
        return [
            l2w[4 * 0 + 0] * pnt[0] + l2w[4 * 1 + 0] * pnt[1] + l2w[4 * 2 + 0] * pnt[2] + l2w[4 * 3 + 0],
            l2w[4 * 0 + 1] * pnt[0] + l2w[4 * 1 + 1] * pnt[1] + l2w[4 * 2 + 1] * pnt[2] + l2w[4 * 3 + 1],
            l2w[4 * 0 + 2] * pnt[0] + l2w[4 * 1 + 2] * pnt[1] + l2w[4 * 2 + 2] * pnt[2] + l2w[4 * 3 + 2],
        ]
    }

    let min = [ 1e30,  1e30,  1e30]
    let max = [-1e30, -1e30, -1e30]

    for (var i = 0; i < 8; i++) {
        let pnt = transformPoint(points[i])

        min = [Math.min(min[0], pnt[0]), Math.min(min[1], pnt[1]), Math.min(min[2], pnt[2])]
        max = [Math.max(max[0], pnt[0]), Math.max(max[1], pnt[1]), Math.max(max[2], pnt[2])]
    }

    return { min, max }
}

function computeTransformMatrices(transform) {
    // compute the individual local -> world transforms
    let l2wr = rotateMat4(   transform.rotation)
    let l2wt = translateMat4(transform.position)
    let l2ws = scaleMat4(    transform.scale   )

    // compute the individual world -> local transforms
    let w2lr = invRotateMat4([ transform.rotation[0],  transform.rotation[1],  transform.rotation[2]])
    let w2lt = translateMat4([-transform.position[0], -transform.position[1], -transform.position[2]])
    let w2ls = scaleMat4(    [ 1./transform.scale[0],  1./transform.scale[1],  1./transform.scale[2]])

    return {
        localToWorld: multiplyMat4(l2wt, multiplyMat4(l2wr, l2ws)),
        worldToLocal: multiplyMat4(w2ls, multiplyMat4(w2lr, w2lt))
    }

    function multiplyMat4(a, b) {
        return [
            a[4 * 0 + 0] * b[4 * 0 + 0] + a[4 * 1 + 0] * b[4 * 0 + 1] + a[4 * 2 + 0] * b[4 * 0 + 2] + a[4 * 3 + 0] * b[4 * 0 + 3],
            a[4 * 0 + 1] * b[4 * 0 + 0] + a[4 * 1 + 1] * b[4 * 0 + 1] + a[4 * 2 + 1] * b[4 * 0 + 2] + a[4 * 3 + 1] * b[4 * 0 + 3],
            a[4 * 0 + 2] * b[4 * 0 + 0] + a[4 * 1 + 2] * b[4 * 0 + 1] + a[4 * 2 + 2] * b[4 * 0 + 2] + a[4 * 3 + 2] * b[4 * 0 + 3],
            a[4 * 0 + 3] * b[4 * 0 + 0] + a[4 * 1 + 3] * b[4 * 0 + 1] + a[4 * 2 + 3] * b[4 * 0 + 2] + a[4 * 3 + 3] * b[4 * 0 + 3],
            a[4 * 0 + 0] * b[4 * 1 + 0] + a[4 * 1 + 0] * b[4 * 1 + 1] + a[4 * 2 + 0] * b[4 * 1 + 2] + a[4 * 3 + 0] * b[4 * 1 + 3],
            a[4 * 0 + 1] * b[4 * 1 + 0] + a[4 * 1 + 1] * b[4 * 1 + 1] + a[4 * 2 + 1] * b[4 * 1 + 2] + a[4 * 3 + 1] * b[4 * 1 + 3],
            a[4 * 0 + 2] * b[4 * 1 + 0] + a[4 * 1 + 2] * b[4 * 1 + 1] + a[4 * 2 + 2] * b[4 * 1 + 2] + a[4 * 3 + 2] * b[4 * 1 + 3],
            a[4 * 0 + 3] * b[4 * 1 + 0] + a[4 * 1 + 3] * b[4 * 1 + 1] + a[4 * 2 + 3] * b[4 * 1 + 2] + a[4 * 3 + 3] * b[4 * 1 + 3],
            a[4 * 0 + 0] * b[4 * 2 + 0] + a[4 * 1 + 0] * b[4 * 2 + 1] + a[4 * 2 + 0] * b[4 * 2 + 2] + a[4 * 3 + 0] * b[4 * 2 + 3],
            a[4 * 0 + 1] * b[4 * 2 + 0] + a[4 * 1 + 1] * b[4 * 2 + 1] + a[4 * 2 + 1] * b[4 * 2 + 2] + a[4 * 3 + 1] * b[4 * 2 + 3],
            a[4 * 0 + 2] * b[4 * 2 + 0] + a[4 * 1 + 2] * b[4 * 2 + 1] + a[4 * 2 + 2] * b[4 * 2 + 2] + a[4 * 3 + 2] * b[4 * 2 + 3],
            a[4 * 0 + 3] * b[4 * 2 + 0] + a[4 * 1 + 3] * b[4 * 2 + 1] + a[4 * 2 + 3] * b[4 * 2 + 2] + a[4 * 3 + 3] * b[4 * 2 + 3],
            a[4 * 0 + 0] * b[4 * 3 + 0] + a[4 * 1 + 0] * b[4 * 3 + 1] + a[4 * 2 + 0] * b[4 * 3 + 2] + a[4 * 3 + 0] * b[4 * 3 + 3],
            a[4 * 0 + 1] * b[4 * 3 + 0] + a[4 * 1 + 1] * b[4 * 3 + 1] + a[4 * 2 + 1] * b[4 * 3 + 2] + a[4 * 3 + 1] * b[4 * 3 + 3],
            a[4 * 0 + 2] * b[4 * 3 + 0] + a[4 * 1 + 2] * b[4 * 3 + 1] + a[4 * 2 + 2] * b[4 * 3 + 2] + a[4 * 3 + 2] * b[4 * 3 + 3],
            a[4 * 0 + 3] * b[4 * 3 + 0] + a[4 * 1 + 3] * b[4 * 3 + 1] + a[4 * 2 + 3] * b[4 * 3 + 2] + a[4 * 3 + 3] * b[4 * 3 + 3],
        ]
    }

    function translateMat4(pos) {
        return [
               1.0,    0.0,    0.0, 0.0,
               0.0,    1.0,    0.0, 0.0,
               0.0,    0.0,    1.0, 0.0,
            pos[0], pos[1], pos[2], 1.0,
        ]
    }

    function scaleMat4(s) {
        return [
            s[0],  0.0,  0.0, 0.0,
             0.0, s[1],  0.0, 0.0,
             0.0,  0.0, s[2], 0.0,
             0.0,  0.0,  0.0, 1.0,
        ]
    }

    function rotateMat4(rot) {
        let rx, ry, rz

        {
            let c = Math.cos(rot[0])
            let s = Math.sin(rot[0])
            rx = [
                1., 0., 0., 0.,
                0.,  c,  s, 0.,
                0., -s,  c, 0.,
                0., 0., 0., 1.,
            ]
        }

        {
            let c = Math.cos(rot[1])
            let s = Math.sin(rot[1])
            ry = [
                 c, 0., -s, 0.,
                0., 1., 0., 0.,
                 s, 0.,  c, 0.,
                0., 0., 0., 1.,
            ]
        }

        {
            let c = Math.cos(rot[2])
            let s = Math.sin(rot[2])
            rz = [
                 c,  s, 0., 0.,
                -s,  c, 0., 0.,
                0., 0., 1., 0.,
                0., 0., 0., 1.
            ]
        }

        return multiplyMat4(rz, multiplyMat4(ry, rx))
    }

    function invRotateMat4(rot) {
        let rx, ry, rz

        {
            let c = Math.cos(-rot[0])
            let s = Math.sin(-rot[0])
            rx = [
                1., 0., 0., 0.,
                0.,  c,  s, 0.,
                0., -s,  c, 0.,
                0., 0., 0., 1.,
            ]
        }

        {
            let c = Math.cos(-rot[1])
            let s = Math.sin(-rot[1])
            ry = [
                 c, 0., -s, 0.,
                0., 1., 0., 0.,
                 s, 0.,  c, 0.,
                0., 0., 0., 1.,
            ]
        }

        {
            let c = Math.cos(-rot[2])
            let s = Math.sin(-rot[2])
            rz = [
                 c,  s, 0., 0.,
                -s,  c, 0., 0.,
                0., 0., 1., 0.,
                0., 0., 0., 1.
            ]
        }

        return multiplyMat4(rx, multiplyMat4(ry, rz))
    }
}