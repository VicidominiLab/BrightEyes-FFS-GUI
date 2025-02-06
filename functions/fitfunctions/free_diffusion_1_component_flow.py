from brighteyes_ffs.fcs.fcs_fit import fcs_fit_dualfocus


def get_params():
    return {
        "model"                : 'Free diffusion 1 component with flow',
        "shortlabel"           : 'Free diff 1 comp flow',
        "paramNames"           : ["N", "D (µm^2/s)", "Beam waist (nm)", "Shape parameter", "rho x (nm)", "rho y (nm)", "vx (nm/ms)", "vy (nm/ms)", "Offset"],
        "paramFittable"        : [True, True, True, True, True, True, True, True, True],
        "paramDefvalues"       : [1, 1, 250, 3, 150, 150, 0, 0, 0],
        # [c, tauD, w2 for all, SF for all, rho for all, vx, vy, offset for all]
        "allparamDefvalues"    : [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        "paramFactors10"       : [1, 1, 1, 1, 1, 1, 1e3, 1e3, 1],
        "paramMinbound"        : [0, 5e-2, 5e-2, 0, -1e6, -1e6, -1e6, -1e6, -1e2],
        "paramMaxbound"        : [1e6, 1000, 1e4, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6],
        "fitfunctionName"      : fcs_fit_dualfocus,
        "fitfunctionParamUsed" : [0, 1, 2, 3, 4, 5, 6, 7, 8]
    }
