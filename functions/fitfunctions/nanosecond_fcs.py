from brighteyes_ffs.fcs.fcs_analytical import nanosecond_fcs_analytical

def get_params():
    return {
        "model"                : 'Nanosecond FCS',
        "shortlabel"           : 'nanosec_fcs',
        "paramNames"           : ["A", "A antibunching", "tau antibunching (us)", "A conformational", "tau conform. (us)", "A rotational", "tau rot. (us)", "A triplet", "tau triplet (us)", "tau D (ms)", "SP"],
        "paramFittable"        : [True, True, True, True, True, True, True, True, True, True, True],
        "paramDefvalues"       : [1, 1, 10, 1, 50, 1, 100, 1, 200, 1, 3],
        "paramFactors10"       : [1, 1, 1e-6, 1, 1e-6, 1, 1e-6, 1, 1e-6, 1e-3, 1],
        
                                # A, c_ab, tau_ab, c_conf, tau_conf, c_rot, tau_rot, c_trip, tau_trip, tauD, SP
        "allparamDefvalues"    : [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        "paramMinbound"        : [0, 0, 1e-5, 0, 1e-5, 0, 1e-5, 0, 1e-5, 1e-5, 1e-6],
        "paramMaxbound"        : [1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6],
        "fitfunctionName"      : nanosecond_fcs_analytical,
        "fitfunctionParamUsed" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
