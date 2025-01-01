
function getLambertDiffuseBRDF() {
    return /* wgsl */ `
    fn lambert_diffuse_sample_f(
        wo : vec3f, 
        wi : ptr<function, vec3f>, 
        seed : ptr<function, f32>, 
        albedo : vec3f,
        flags : ptr<function, u32>
    ) -> vec4f {
        *wi = cosineSampleHemisphere(rand2(*seed)); *seed += 2.f;
        return vec4f(pow(albedo, vec3f(2.2)) * InvPi, (*wi).z);
    }

    fn lambert_diffuse_f(
        wo : vec3f,
        wi : vec3f,
        albedo : vec3f
    ) -> vec3f {
        return pow(albedo, vec3f(2.2)) * InvPi;
    }`
}

function getPerfectMirrorBRDF() {
    return /* wgsl */ `
    fn perfect_mirror_sample_f(
        wo : vec3f, 
        wi : ptr<function, vec3f>, 
        albedo : vec3f,
        flags : ptr<function, u32>
    ) -> vec4f {
        *wi = wo * vec3f(-1.f, -1.f, 1.f);
        *flags |= 1u << 3u;
        return vec4f(pow(albedo, vec3f(2.2)), 1.f);
    }

    fn perfect_mirror_f(
        wo : vec3f,
        wi : vec3f,
        albedo : vec3f
    ) -> vec3f {
        return vec3f(0.f);
    }
    `
}

function getEmissiveBRDF() {
    return /* wgsl */ `
    fn emissive_sample_f(
        wo : vec3f, 
        wi : ptr<function, vec3f>, 
        seed : ptr<function, f32>, 
        emission : vec3f,
        flags : ptr<function, u32>
    ) -> vec4f {
        *wi = vec3f(0.f);
        *flags = 2u;
        return vec4f(emission, -1.f);
    }

    fn emissive_f(
        wo : vec3f,
        wi : vec3f,
        emission : vec3f,
    ) -> vec3f {
        return vec3f(0.f);
    }
    `
}