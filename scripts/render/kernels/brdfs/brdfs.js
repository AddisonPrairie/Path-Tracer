
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

function getRoughMirrorBRDF() {
    return /* wgsl */ `

    // based on: https://schuttejoe.github.io/post/ggximportancesamplingpart1/
    fn schlick_fresnel(r0 : vec3f, radians : f32) -> vec3f {
        var exponent : f32 = pow(1.f - radians, 5.f);
        return r0 + (1.f - r0) * exponent;
    }

    fn ggx_d(
        wm : vec3f,
        a2 : f32
    ) -> f32 {
        var denom : f32 = ((wm.z * wm.z) * (a2 - 1.f) + 1.f);
        denom = Pi * (denom * denom);
        return a2 / denom;
    }

    fn vndf_pdf(
        wo : vec3f,
        wm : vec3f,
        a2 : f32
    ) -> f32 {
        var wowm : f32 = abs(dot(wo, wm));

        var wi : vec3f = reflect(-wo, wm);

        var G1 : f32 = smith_ggx_masking(wi, wo, a2);
        var D  : f32 = ggx_d(wm, a2);

        return G1 * D * wowm / (abs(wo.z));
    }

    fn smith_ggx_masking_shadowing(wi : vec3f, wo : vec3f, a2 : f32) -> f32 {
        var dot_nl : f32 = wi.z;
        var dot_nv : f32 = wo.z;

        var denom_a : f32 = dot_nv * sqrt(a2 + (1.f - a2) * dot_nl * dot_nl);
        var denom_b : f32 = dot_nl * sqrt(a2 + (1.f - a2) * dot_nv * dot_nv);

        return 2.f * dot_nl * dot_nv / (denom_a + denom_b);
    }

    fn smith_ggx_masking(wi : vec3f, wo : vec3f, a2 : f32) -> f32 {
        var dot_nv : f32 = wo.z;
        var denom_c : f32 = sqrt(a2 + (1.f - a2) * dot_nv * dot_nv) + dot_nv;
        return 2.f * dot_nv / denom_c;
    }

    fn ggx_vndf(
        wo : vec3f,
        roughness : f32,
        r2 : vec2f
    ) -> vec3f {
        var v : vec3f = normalize(wo * vec3f(roughness, roughness, 1.f));

        var t1 : vec3f;
        if (v.z < .999f) {
            t1 = normalize(cross(v, vec3f(0.f, 0.f, 1.f)));
        } else {
            t1 = vec3f(1.f, 0.f, 0.f);
        }
        var t2 : vec3f = cross(t1, v);

        var a : f32 = 1.f / (1.f + v.z);
        var r : f32 = sqrt(r2.x);
        var phi : f32;
        if (r2.y < a) {
            phi = (r2.y / a) * Pi;
        } else {
            phi = Pi + (r2.y - a) / (1.f - a) * Pi;
        }
        var p1 : f32 = r * cos(phi);
        var p2 : f32 = r * sin(phi);
        if (r2.y >= a) {
            p2 *= v.z;
        }

        var n : vec3f = p1 * t1 + p2 * t2 + max(0.f, 1. - p1 * p1 - p2 * p2) * v;

        return normalize(n * vec3f(roughness, roughness, 1.f));
    }

    fn ggxd_sample_f(
        r2 : vec2f,
        wo : vec3f,
        wi : ptr<function, vec3f>,
        albedo : vec3f,
        roughness : f32
    ) -> vec4f {
        var a : f32 = roughness;
        var a2 : f32 = a * a;

        var wm : vec3f = ggx_vndf(wo, roughness, r2);

        *wi = reflect(-wo, wm);

        if ((*wi).z > 0.f) {
            var F : vec3f = schlick_fresnel(pow(albedo, vec3f(2.2)), dot(*wi, wm));
            var G1 : f32 = smith_ggx_masking(*wi, wo, a2);
            var G2 : f32 = smith_ggx_masking_shadowing(*wi, wo, a2);

            var pdf : f32 = vndf_pdf(wo, wm, a2);

            // I think there is something wrong about the way the pdf is computed
            return vec4f(F * (G2 / G1) * (*wi).z, pdf);
        } else {
            return vec4f(vec3f(0.f), 1.);
        }
    }

    fn ggxd_f(
        wo : vec3f,
        wi : vec3f,
        albedo : vec3f,
        roughness : f32
    ) -> vec3f {
        // there also seems to be something wrong with this
        var a2 : f32 = roughness * roughness;
        var wm : vec3f = normalize((wo + wi) * .5f);
        var F : vec3f = schlick_fresnel(pow(albedo, vec3f(2.2)), dot(wi, wm));
        var G2 : f32 = smith_ggx_masking_shadowing(wi, wo, a2);
        var D : f32 = ggx_d(wm, a2);

        return F * G2 * D / (4.f * abs(wi.z) * abs(wo.z));
    }

    /*
    
    fn ggx_sample_wm(wo : vec3f, u : vec2f, roughness : f32) -> vec3f {
        var v : vec3f = normalize(wo * vec3f(roughness, roughness, 1.));

        var lensq : f32 = dot(v.xy, v.xy);
        var o1 : vec3f;
        if (lensq > 0.) {
            o1 = normalize(vec3f(-v.y, v.x, 0.));
        } else {
            o1 = vec3f(1., 0., 0.);
        }
        var o2 : vec3f = cross(v, o1);

        var r : f32 = sqrt(u.x);
        var phi : f32 = 2. * Pi * u.y;

        var t1 = r * cos(phi);
        var t2 = r * sin(phi);

        //var s = .5 * (1. + v.z);
        //t2 = (1. - s) * sqrt(1. - t1 * t1) + s * t2;

        var n : vec3f = t1 * o1 + t2 * o2 + sqrt(max(0., 1. - t1 * t1 - t2 * t2)) * v;

        return normalize(vec3f(n.x, n.y, max(n.z, 0.)) * vec3f(roughness, roughness, 1.));
    }

    fn ggx_sample_f(
        wo : vec3f, 
        wi : ptr<function, vec3f>, 
        seed : ptr<function, f32>, 
        albedo : vec3f,
        roughness : f32,
        flags : ptr<function, u32>
    ) -> vec4f {
        var hw : vec3f = ggx_sample_wm(wo, rand2(*seed), roughness); *seed += 2.f;
        *wi = reflect(-wo, hw);

        var refl : vec3f = vec3f(0.f);
        var  pdf :   f32 = ggx_pdf(wo, hw, roughness) / (4.f * dot(hw, wo));


        var G : f32 = 1.f / (1.f + ggx_lambda(wo, roughness) + ggx_lambda(*wi, roughness));
        var F : f32 = 1.f;
        refl = pow(albedo, vec3f(2.2)) * F * G / ggx_G1(wo, roughness);

        if (any(refl != refl) || pdf != pdf || (*wi).z <= 0.) {refl = vec3f(0.);}

        return vec4f(refl, pdf);
    }

    fn ggx_G1(v : vec3f, roughness : f32) -> f32 {
        return 2. / (1. + sqrt(1. + roughness * roughness * dot(v.xy, v.xy) / (v.z * v.z)));
    }

    fn ggx_pdf(v : vec3f, n : vec3f, roughness : f32) -> f32 {
        return ggx_G1(v, roughness) * max(0., dot(v, n)) * ggx_D(n, roughness) / v.z;
    }

    fn ggx_lambda(w : vec3f, alpha : f32) -> f32 {
        var tan2_theta : f32 = (1.0 - w.z * w.z) / (w.z * w.z);
        return (sqrt(1 + alpha * alpha * tan2_theta) - 1.f) / 2.f;
    }

    fn ggx_D(n : vec3f, roughness : f32) -> f32 {
        var a2 : f32 = roughness * roughness;
        var denom : f32 = (dot(n.xy, n.xy) / a2 + n.z * n.z);

        return 1. / (Pi * a2 * denom * denom);
    }
        
    */`
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