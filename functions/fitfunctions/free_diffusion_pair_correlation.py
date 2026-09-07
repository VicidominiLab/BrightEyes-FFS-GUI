from brighteyes_ffs.fcs.fcs_analytical import fcs_dualfocus

def get_params():
    return {
        "model"                : 'Free diffusion pair-correlation',
        "shortlabel"           : 'Free diff pair-corr',
        "paramNames"           : ["N", "D (um^2/s)", "Beam waist (nm)", "Shape parameter", "rho (nm)", "Offset"],
        "paramFittable"        : [True, True, True, True, True, True],
        "paramDefvalues"       : [1, 1, 600, 3, 150, 0],
        "paramFactors10"       : [1, 1e-12, 1e-9, 1, 1e-9, 1],
        
                               # [N, D, w, SF, rhox, rhoy, offset, vx, vy
        "allparamDefvalues"    : [-1, -1, -1, -1, -1, 0, -1, 0, 0] ,
        "paramMinbound"        : [1e-3, 1e-3, 1, 0.1, 0, 0, -1e6, -1e6, -1e6],
        "paramMaxbound"        : [1e6, 1e4, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6],
        "fitfunctionName"      : fcs_dualfocus,
        "fitfunctionParamUsed" : [0, 1, 2, 3, 4, 6]
    }
