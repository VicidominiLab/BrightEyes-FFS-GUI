from brighteyes_ffs.fcs.fcs_fit import fitfun_finitelength

def get_params():
    return {
        "model"                : 'Free diffusion 1 component - finite length',
        "shortlabel"           : 'Free diff 1 comp finite length',
        "paramNames"           : ["N", "Tau (ms)", "Shape parameter", "T (s)", "Tsampling (ms)", "Brightness (Hz)"],
        "paramFittable"        : [True, True, True, False, False, True],
        "paramDefvalues"       : [1, 1, 3, 1, 0.001, 100],
        # N, tau1, tau2, F, alpha, T, tautrip, SP, offset, A, B
        "allparamDefvalues"    : [-1, -1, -1, -1, -1, -1],
        "paramFactors10"       : [1, 1, 1, 1, 1, 1],
        "paramMinbound"        : [0, 1e-3, 1e-3, 1e-9, 1e-6, 1e-6],
        "paramMaxbound"        : [1e6, 1000, 1000, 1e9, 1e6, 1e9],
        "fitfunctionName"      : fitfun_finitelength,
        "fitfunctionParamUsed" : [0, 1, 2, 3, 4, 5]
    }
