from brighteyes_ffs.fcs.fcs_analytical import fcs_2c_analytical


def get_params():
    return {
        "model"                : 'Free diffusion 2 components',
        "shortlabel"           : 'Free diff 2 comp',
        "paramNames"           : ["N", "Tau 1 (ms)", "Tau 2 (ms)", "Fraction species 1", "Rel.brightness 1", "Triplet fraction", "Triplet time (µs)", "Shape parameter", "Offset"],
        "paramFittable"        : [True, True, True, True, True, True, True, True, True],
        "paramDefvalues"       : [1, 0.1, 1, 0.5, 1, 0.1, 10, 3, 0],
        "paramFactors10"       : [1, 1e-3, 1e-3, 1, 1, 1, 1e-6, 1, 1],
        
                                # N, tau1, tau2, F, alpha, T, tautrip, SP, offset, A, B
        "allparamDefvalues"    : [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1.05],
        "paramMinbound"        : [0, 1e-3, 1e-3, 0, 0, 0, 0, 0, -1e2, 0, 0],
        "paramMaxbound"        : [1e6, 1000, 1000, 1, 1e6, 1, 1e6, 1e6, 1e6, 1e6, 1e6],
        "fitfunctionName"      : fcs_2c_analytical,
        "fitfunctionParamUsed" : [0, 1, 2, 3, 4, 5, 6, 7, 8]
    }
