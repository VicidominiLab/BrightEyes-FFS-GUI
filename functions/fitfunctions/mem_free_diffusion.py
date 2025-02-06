from brighteyes_ffs.fcs.mem_fit import mem_fit_free_diffusion

def get_params():
    return {
        "model"                : 'Maximum entropy method free diffusion',
        "shortlabel"           : 'MEM free diff K',
        "paramNames"           : ["Ncomp", "Corr vs. entropy", "Shape parameter"],
        "paramFittable"        : [False, False, False],
        "paramDefvalues"       : [200, 20, 3],
        # Ncomp, Niter, eps, xmu, anomD, corrVentr
        "allparamDefvalues"    : [-1, 10000, 5e-6, 2e-4, 1, -1, -1],
        "paramFactors10"       : [1, 1, 1, 1, 1, 1, 1],
        "paramMinbound"        : [0, 0, 0, 0, 0, 0, 0],
        "paramMaxbound"        : [1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6],
        "fitfunctionName"      : mem_fit_free_diffusion,
        "fitfunctionParamUsed" : [0, 5, 6]
    }
