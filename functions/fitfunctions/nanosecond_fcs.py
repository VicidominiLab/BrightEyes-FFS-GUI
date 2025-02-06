from brighteyes_ffs.fcs.fcs_fit import fitfun_nanosecond_fcs

def get_params():
    return {
        "model"                : 'Nanosecond FCS',
        "shortlabel"           : 'nanosec_fcs',
        "paramNames"           : ["A", "A antibunching", "tau antibunching (us)", "A conformational", "tau conform. (us)", "A rotational", "tau rot. (us)", "A triplet", "tau triplet (us)", "tau D (ms)", "SP"],
        "paramFittable"        : [True, True, True, True, True, True, True, True, True, True, True],
        "paramDefvalues"       : [1, 1, 10, 1, 50, 1, 100, 1, 200, 1, 3],
        # N, tau1, tau2, F, alpha, T, tautrip, SP, offset, A, B
        "allparamDefvalues"    : [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        "paramFactors10"       : [1, 1, 1e-6, 1, 1e-6, 1, 1e-6, 1, 1e-6, 1e-3, 1],
        "paramMinbound"        : [0, 0, 1e-5, 0, 1e-5, 0, 1e-5, 0, 1e-5, 1e-5, 1e-6],
        "paramMaxbound"        : [1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6],
        "fitfunctionName"      : fitfun_nanosecond_fcs,
        "fitfunctionParamUsed" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
