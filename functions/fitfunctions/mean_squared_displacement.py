def get_params():
    return {
        "model"                : 'Mean squared displacement',
        "shortlabel"           : 'iMSD',
        "paramNames"           : ['D (um^2/s)', 'Offset', 'Smoothing', 'Px dist (nm)'],
        "paramFittable"        : [True, True, False, False],
        "paramDefvalues"       : [1, 1, 1, 1],
        # N, tau1, tau2, F, alpha, T, tautrip, SP, offset, A, B
        "allparamDefvalues"    : [-1, -1, -1, -1],
        "paramFactors10"       : [1, 1, 1, 1],
        "paramMinbound"        : [-1e9, -1e9, -1e9, -1e9],
        "paramMaxbound"        : [1e9, 1e9, 1e9, 1e9],
        "fitfunctionName"      : None,
        "fitfunctionParamUsed" : [0, 1, 2, 3]
    }
