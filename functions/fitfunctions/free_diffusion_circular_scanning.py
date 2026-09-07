from brighteyes_ffs.fcs.fcs_analytical import fcs_circular_scanning

def get_params():
    return {
        "model"                : 'Free diffusion circular scanning',
        "shortlabel"           : 'Free diff circ',
        "paramNames"           : ["N", "Tau (ms)", "Beam waist (nm)", "Shape parameter", "Circle radius (nm)", "Circle period (µs)", "Offset/1000"],
        "paramFittable"        : [True, True, True, True, True, True, True],
        "paramDefvalues"       : [1, 1, 600, 3, 500, 320, 0],
        "paramFactors10"       : [1, 1e-3, 1e-9, 1, 1e-9, 1e-6, 1],
        
                                # N, tau_D, w, SF, orbit_time, orbit_radius, offset, vx, vy
        "allparamDefvalues"    : [-1, -1, -1, -1, -1, -1, -1, 0, 0],
        "paramMinbound"        : [0, 1e-9, 0, 0, 0, 0, -1e6, 0, 0],
        "paramMaxbound"        : [1e10, 1e10, 1e10, 100, 1e10, 1e10, 1e10, 1e10, 1e10],
        "fitfunctionName"      : fcs_circular_scanning,
        "fitfunctionParamUsed" : [0, 1, 2, 3, 5, 4, 6]
    }
