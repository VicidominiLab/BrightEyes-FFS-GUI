from brighteyes_ffs.fcs.fcs_analytical import fcs_dualfocus_c


def get_params():
    return {
        "model"                : 'Free diffusion 1 component with flow',
        "shortlabel"           : 'Free diff 1 comp flow',
        "paramNames"           : ["c (/µm^3)", "D (µm^2/s)", "Beam waist (nm)", "Shape parameter", "rho x (nm)", "rho y (nm)", "vx (nm/ms)", "vy (nm/ms)", "Offset"],
        "paramFittable"        : [True, True, True, True, True, True, True, True, True],
        "paramDefvalues"       : [1, 1, 250, 3, 150, 150, 0, 0, 0],
        
                                # c, D, w, SF, rhox, rhoy, offset, vx, vy
        "paramFactors10"       : [1e18, 1e-12, 1e-9, 1, 1e-9, 1e-9, 1, 1e-6, 1e-6],
        "allparamDefvalues"    : [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        "paramMinbound"        : [0, 1e-30, 5e-2, 0, -1e30, -1e30, -1e6, -1e18, -1e18],
        "paramMaxbound"        : [1e30, 1e4, 1e4, 1e4, 1e30, 1e30, 1e6, 1e18, 1e18],
        "fitfunctionName"      : fcs_dualfocus_c,
        "fitfunctionParamUsed" : [0, 1, 2, 3, 4, 5, 7, 8, 6]
    }
