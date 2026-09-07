from brighteyes_ffs.fcs.fcs_analytical import fcs_2c_2d_analytical

def get_params():
    return {
        "model"                : '2D free diffusion 1 component',
        "shortlabel"           : '2D free diff 1 comp',
        "paramNames"           : ["N", "Tau 1 (ms)", "Offset"],
        "paramFittable"        : [True, True, True],
        "paramDefvalues"       : [1, 1, 0],
        "paramFactors10"       : [1, 1e-3, 1],
        # N, tau1, tau2, F, alpha, T, tautrip, offset, A, B
        "allparamDefvalues"    : [-1, -1, 0.8e-10, 1, 1, 0, 1e-6, -1, 0, 1.05],
        "paramMinbound"        : [0, 1e-3, 0, 0, 0, 0, 0, -1e2, 0, 0],
        "paramMaxbound"        : [1e6, 1000, 1000, 1e6, 1000, 1e6, 1e6, 1e6, 1e6, 1e6],
        "fitfunctionName"      : fcs_2c_2d_analytical,
        "fitfunctionParamUsed" : [0, 1, 7]
    }
