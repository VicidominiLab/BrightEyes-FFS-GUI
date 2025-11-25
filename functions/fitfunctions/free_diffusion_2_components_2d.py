from brighteyes_ffs.fcs.fcs_fit import fitfun_free_diffusion_2d


def get_params():
    return {
        "model"                : '2D free diffusion 2 components',
        "shortlabel"           : '2D free diff 1 comp',
        "paramNames"           : ["N", "Tau 1 (ms)", "Tau 2 (ms)", "Fraction species 1", "Rel.brightness 1", "Triplet fraction", "Triplet time (µs)", "Offset", "A", "B"],
        "paramFittable"        : [True, True, True, True, True, True, True, True, True, True],
        "paramDefvalues"       : [1, 0.1, 1, 0.5, 1, 0, 1, 0, 1, 1],
        # N, tau1, tau2, F, alpha, T, tautrip, SP, offset, A, B
        "allparamDefvalues"    : [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        "paramFactors10"       : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "paramMinbound"        : [0, 1e-3, 5e-2, 0, 0, 0, 0, -1e2, 0, 0],
        "paramMaxbound"        : [1e6, 1000, 1000, 1, 1e6, 1, 1e6, 1e6, 1e6, 1e6],
        "fitfunctionName"      : fitfun_free_diffusion_2d,
        "fitfunctionParamUsed" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
