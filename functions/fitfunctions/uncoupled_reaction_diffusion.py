from brighteyes_ffs.fcs.fcs_fit import fitfun_uncoupled_reaction_diffusion

def get_params():
    return {
        "model"                : 'Uncoupled reaction and diffusion',
        "shortlabel"           : 'reaction_diffusion',
        "paramNames"           : ["A", "Tau (ms)", "Shape parameter", "Fraction free", "k_off (/s)"],
        "paramFittable"        : [True, True, True, True, True],
        "paramDefvalues"       : [1, 1, 3, 0.99, 1],
        # tau, A, tauD, SP, f_eq, k_off
        "allparamDefvalues"    : [-1, -1, -1, -1, -1],
        "paramFactors10"       : [1, 1e-3, 1, 1, 1],
        "paramMinbound"        : [0, 1e-9, 1e-9, 0, 1e-9],
        "paramMaxbound"        : [1e6, 1e6, 1e6, 1, 1e6],
        "fitfunctionName"      : fitfun_uncoupled_reaction_diffusion,
        "fitfunctionParamUsed" : [0, 1, 2, 3, 4]
    }

