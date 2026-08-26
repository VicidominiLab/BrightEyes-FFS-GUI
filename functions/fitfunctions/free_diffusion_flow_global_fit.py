from brighteyes_ffs.fcs.fcs_analytical import fcs_dualfocus_c

def get_params():
    return {
        "model"                : 'Free diffusion with flow - global fit',
        "shortlabel"           : 'Free diff 1 comp flow global',
        "paramNames"           : ["c (/µm^3)", "D (um^2/s)", "Beam waist (nm)", "Shape parameter", "rho x (nm)", "rho y (nm)", "vx (nm/ms)", "vy (nm/ms)", "Offset"],
        "paramFittable"        : [True, True, True, True, True, True, True, True, True],
        "paramDefvalues"       : [1, 1, 250, 3, 150, 150, 0, 0, 0],
        "paramFactors10"       : [1e18, 1e-12, 1e-9, 1, 1e-9, 1e-9, 1e-6, 1e-6, 1],
        
        # c, D, w, SF, rhox, rhoy, offset, vx=0, vy=0)
        "globalParam"          : [True, True, False, False, False, False, False, True, True],
        "allparamDefvalues"    : [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        "paramMinbound"        : [0, 1e-30, 5e-2, 0, -1e10, -1e10, -1e18, -1e18, -1e18],
        "paramMaxbound"        : [1e30, 1e6, 1e6, 1e6, 1e30, 1e30, 1e18, 1e18, 1e18],
        "fitfunctionName"      : fcs_dualfocus_c,
        "fitfunctionParamUsed" : [0, 1, 2, 3, 4, 5, 7, 8, 6]
    }

