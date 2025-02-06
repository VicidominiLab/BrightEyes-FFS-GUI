def get_params():
    return {
        "model"                : 'Model-free displacement analysis',
        "shortlabel"           : 'MFDA',
        "paramNames"           : ['Smoothing'],
        "paramFittable"        : [False],
        "paramDefvalues"       : [5],
        # N, tau1, tau2, F, alpha, T, tautrip, SP, offset, A, B
        "allparamDefvalues"    : [-1],
        "paramFactors10"       : [1],
        "paramMinbound"        : [1],
        "paramMaxbound"        : [1000],
        "fitfunctionName"      : None,
        "fitfunctionParamUsed" : [0]
    }
