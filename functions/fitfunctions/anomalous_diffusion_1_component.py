from brighteyes_ffs.fcs.fcs_analytical import fcs_analytical_2c_anomalous

def get_params():
    return {
        "model"                : 'Anomalous diffusion 1 component',
        "shortlabel"           : 'Anomalous diff 1 comp',
        "paramNames"           : ["N", "Tau (ms)", "Shape parameter", "Offset", "Alpha"],
        "paramFittable"        : [True, True, True, True, True],
        "paramDefvalues"       : [1, 1, 3, 0, 1],
        "paramFactors10"       : [1, 1e-3, 1, 1, 1],
                                # N,  tauD1, tauD2,  alpha1, alpha2, F, T, tau_triplet, SF, offset, brightness):
        "allparamDefvalues"    : [-1, -1,    8e-11, -1,  1,   1, 0, 1e-6,  -1,    -1,   1],
        "paramMinbound"        : [0,   1e-6, 1e-6,  0,   0,   0, 0, 1e-12, 1e-12, -1e6, 0],
        "paramMaxbound"        : [1e6, 1000, 1000,  1e6, 1e6, 1, 1, 1e6,   1e6,   1e6,  1e6],
        "fitfunctionName"      : fcs_analytical_2c_anomalous,
        "fitfunctionParamUsed" : [0, 1, 8, 9, 3]
    }
