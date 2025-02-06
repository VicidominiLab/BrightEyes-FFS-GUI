from brighteyes_ffs.fcs.fcs_fit import fitfun_an_2c

def get_params():
    return {
        "model"                : 'Anomalous diffusion 2 components',
        "shortlabel"           : 'Anom Free diff 2 comp',
        "paramNames"           : ["N", "Tau 1 (ms)", "Tau 2 (ms)", "Alpha 1", "Alpha 2", "Fraction species 1", "Triplet fraction", "Triplet time (µs)", "Shape parameter", "Offset"],
        "paramFittable"        : [True, True, True, True, True, True, True, True, True, True],
        "paramDefvalues"       : [1, 1, 10, 1, 1, 0.5, 0.1, 10, 3, 0],
        #  N, tauD1, tauD2, alpha1, alpha2, F, T, tau_triplet, SF, offset
        "allparamDefvalues"    : [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        "paramFactors10"       : [1, 1e-3, 1e-3, 1, 1, 1, 1, 1e-6, 1, 1],
        "paramMinbound"        : [0, 1e-6, 1e-6, 0, 0, 0, 0, 0, 0, -5e6],
        "paramMaxbound"        : [1e6, 1e6, 1e6, 1e6, 1e6, 1, 1, 1e6, 1e6, 1e6],
        "fitfunctionName"      : fitfun_an_2c,
        "fitfunctionParamUsed" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    