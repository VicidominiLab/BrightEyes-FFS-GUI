from brighteyes_ffs.fcs.fcs_fit import fitfun_2c

def get_params():
    return {
        "model"                : 'Free diffusion 1 component',
        "shortlabel"           : 'Free diff 1 comp',
        "paramNames"           : ["N", "Tau (ms)", "Shape parameter", "Offset"],
        "paramFittable"        : [True, True, True, True],
        "paramDefvalues"       : [1, 1, 3, 0],
        # N, tau1, tau2, F, alpha, T, tautrip, SP, offset, A, B
        "allparamDefvalues"    : [-1, -1, 0.8e-10, 1, 1, 0, 1e-6, -1, -1, 0, 1.05],
        "paramFactors10"       : [1, 1, 1, 1],
        "paramMinbound"        : [0, 1e-2, 5e-2, 0, 0, 0, 0, 0, -1e2, 0, 0],
        "paramMaxbound"        : [1e6, 1000, 1000, 1, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6],
        "fitfunctionName"      : fitfun_2c,
        "fitfunctionParamUsed" : [0, 1, 7, 8]
    }