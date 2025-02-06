from brighteyes_ffs.fcs.fcs_fit import fitfun_an

def get_params():
    return {
        "model"                : 'Anomalous diffusion 1 component',
        "shortlabel"           : 'Anomalous diff 1 comp',
        "paramNames"           : ["N", "Tau (ms)", "Shape parameter", "Offset", "Alpha"],
        "paramFittable"        : [True, True, True, True, True],
        "paramDefvalues"       : [1, 1, 3, 0, 1],
        # N, tau1, tau2, F, alpha, T, tautrip, SP, offset, A, B
        "allparamDefvalues"    : [-1, -1, -1, -1, -1],
        "paramFactors10"       : [1, 1e-3, 1, 1, 1],
        "paramMinbound"        : [0, 1e-5, 5e-2, 0, 0],
        "paramMaxbound"        : [1e6, 1000, 1000, 1e6, 1000],
        "fitfunctionName"      : fitfun_an,
        "fitfunctionParamUsed" : [0, 1, 2, 3, 4]
    }
    