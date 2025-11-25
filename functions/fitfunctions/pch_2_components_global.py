from brighteyes_ffs.pch.pch_fit import fitfun_pch_nc_global

def get_params():
    return {
        "model"                : 'PCH 2 components - global fit',
        "shortlabel"           : 'pch 2 comp',
        "paramNames"           : ["Conc. 1 (/um^3)", "Conc. 2 (/um^3)", "Brightness 1 (kHz)", "Brightness 2 (kHz)", 'Dark count rate (kHz)', "Bin time (ms)", 'w0 (nm)', 'z0/w0', 'N bins'],
                                # c1 c2 b1 b2 bg time w0 K Nbins
        "paramFittable"        : [True, True, True, True, True, False, False, False, False],
        "globalParam"          : [True, True, False, False, False, True, False, False, False],
        "paramDefvalues"       : [1, 1, 1, 1, 0, 0.01, 300, 3, 1e4],
        "allparamDefvalues"    : [-1, -1, -1, -1, -1, -1, 2e-6, -1, -1, -1],
        "paramFactors10"       : [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "paramMinbound"        : [1e-12,1e-12,1e-12,1e-12, 0, 1e-12, 1e-12, 1e-12, 1e-12, 10],
        "paramMaxbound"        : [1e12, 1e12, 1e12, 1e12, 1e5, 1e12, 1e12, 1e12, 1e12, 1e12],
        "fitfunctionName"      : fitfun_pch_nc_global,
        "fitfunctionParamUsed" : [0, 1, 2, 3, 4, 5, 7, 8, 9]
    }