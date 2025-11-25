from brighteyes_ffs.fcs.fcs_fit import fitfun_circfcs

def get_params():
    return {
        "model"                : 'Free diffusion circular scanning',
        "shortlabel"           : 'Free diff circ',
        "paramNames"           : ["N", "Tau (ms)", "Beam waist (nm)", "Shape parameter", "Circle radius (nm)", "Circle period (µs)", "Offset/1000"],
        "paramFittable"        : [True, True, True, True, True, True, True],
        "paramDefvalues"       : [1, 1, 600, 3, 500, 320, 0],
        # N, tau1, tau2, F, alpha, T, tautrip, SP, offset, A, B
        "allparamDefvalues"    : [-1, -1, -1, -1, -1, -1, -1],
        "paramFactors10"       : [1, 1, 1e-9, 1, 1e-9, 1e-6, 1e-3],
        "paramMinbound"        : [0, 1e-3, 0, 0, 0, 0, 0],
        "paramMaxbound"        : [1e6, 1000, 10000, 100, 1e6, 1e6, 1e6],
        "fitfunctionName"      : fitfun_circfcs,
        "fitfunctionParamUsed" : [0, 1, 2, 3, 4, 5, 6]
    }
