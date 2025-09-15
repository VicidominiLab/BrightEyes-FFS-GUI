from brighteyes_ffs.pch.pch_fit import fitfun_pch

def get_params():
    return {
        "model"                : 'PCH 1 component',
        "shortlabel"           : 'pch 1 comp',
        "paramNames"           : ["Conc. (/um^3)", "Brightness (kHz)", "Bin time (ms)", 'w0 (nm)', 'z0/w0', 'N bins'],
        "paramFittable"        : [True, True, False, False, False, False],
        "paramDefvalues"       : [1, 1, 1, 300, 3, 1e4],
        # N, tau1, tau2, F, alpha, T, tautrip, SP, offset, A, B
        "allparamDefvalues"    : [-1, -1, -1, 2e-6, -1, -1, -1],
        "paramFactors10"       : [1, 1, 1, 1, 1, 1],
        "paramMinbound"        : [1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 10],
        "paramMaxbound"        : [1e12, 1e12, 1e12, 1e12, 1e12, 1e12, 1e12],
        "fitfunctionName"      : fitfun_pch,
        "fitfunctionParamUsed" : [0, 1, 2, 4, 5, 6]
    }