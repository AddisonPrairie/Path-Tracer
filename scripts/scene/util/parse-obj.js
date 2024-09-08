function parseObj(file) {

    let numTris = 0
    let trisArr = []

    let x_min =  1e30
    let y_min =  1e30
    let z_min =  1e30
    let x_max = -1e30
    let y_max = -1e30
    let z_max = -1e30

    {
        const lines = file.split("\n")
        let vertexPositions = [1e30, 1e30, 1e30]

        for (var x = 0; x < lines.length; x++) {
            const line = lines[x]

            if (line.length <= 1) continue

            if (
                line[0] === "v" || line[0] === "f" || line[0] === "o"
            ) {
                const tokens = line.split(/\s+/)

                switch (tokens[0]) {
                    case "v":
                        let vx = parseFloat(tokens[1])
                        let vy = parseFloat(tokens[2])
                        let vz = parseFloat(tokens[3])

                        x_min = Math.min(x_min, vx)
                        x_max = Math.max(x_max, vx)
                        y_min = Math.min(y_min, vy)
                        y_max = Math.max(y_max, vy)
                        z_min = Math.min(z_min, vz)
                        z_max = Math.max(z_max, vz)

                        vertexPositions.push(vx, vy, vz)
                        break
                    case "f":
                        let idxs = []

                        for (var y = 1; y < tokens.length; y++) {
                            let idx = parseInt(tokens[y])

                            if (!isNaN(idx)) idxs.push(idx)
                        }

                        let vrx = vertexPositions[3 * idxs[0] + 0]
                        let vry = vertexPositions[3 * idxs[0] + 1]
                        let vrz = vertexPositions[3 * idxs[0] + 2]

                        for (var i = 1; i < idxs.length - 1; i++) {
                            let v1x = vertexPositions[3 * idxs[i + 0] + 0]
                            let v1y = vertexPositions[3 * idxs[i + 0] + 1]
                            let v1z = vertexPositions[3 * idxs[i + 0] + 2]
                            let v2x = vertexPositions[3 * idxs[i + 1] + 0]
                            let v2y = vertexPositions[3 * idxs[i + 1] + 1]
                            let v2z = vertexPositions[3 * idxs[i + 1] + 2]
                            
                            numTris++
                            trisArr.push(
                                vrx, vry, vrz, 3.1415,
                                v1x, v1y, v1z, 3.1415,
                                v2x, v2y, v2z, 3.1415
                            )
                        }

                        break
                    case "o":
                        vertexPositions = [1e30, 1e30, 1e30]
                        break
                    default:
                        break
                }
            }
        }
    }

    return { 
        numTriangles: numTris, 
        triangleArray: trisArr, 
        bounds: {
            min: [x_min, y_min, z_min],
            max: [x_max, y_max, z_max]
        } 
    }
}