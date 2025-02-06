from brighteyes_ffs.fcs.fcs_fit import fitfun_free_diffusion_2d

def get_params():
    return {
        "model"                : '2D free diffusion 1 component',
        "shortlabel"           : '2D free diff 1 comp',
        "paramNames"           : ["N", "Tau 1 (ms)", "Offset"],
        "paramFittable"        : [True, True, True],
        "paramDefvalues"       : [1, 1, 0],
        # N, tau1, tau2, F, alpha, T, tautrip, SP, offset, A, B
        "allparamDefvalues"    : [-1, -1, 0.8e-10, 1, 1, 0, 1e-6, -1, 0, 1.05],
        "paramFactors10"       : [1, 1, 1],
        "paramMinbound"        : [0, 1e-2, 0, 0, 0, 0, 0, -1e2, 0, 0],
        "paramMaxbound"        : [1e6, 1000, 1000, 1e6, 1000, 1e6, 1e6, 1e6, 1e6, 1e6],
        "fitfunctionName"      : fitfun_free_diffusion_2d,
        "fitfunctionParamUsed" : [0, 1, 7]
    }
