from brighteyes_ffs.fcs.fcs_analytical import fcs_analytical_2c_anomalous

def get_params():
    return {
        "model"                : 'Anomalous diffusion 2 components',
        "shortlabel"           : 'Anom Free diff 2 comp',
        "paramNames"           : ["N", "Tau 1 (ms)", "Tau 2 (ms)", "Alpha 1", "Alpha 2", "Fraction species 1", "Relative brightness", "Triplet fraction", "Triplet time (µs)", "Shape parameter", "Offset"],
        "paramFittable"        : [True, True, True, True, True, True, True, True, True, True, True],
        "paramDefvalues"       : [1, 1, 10, 1, 1, 0.5, 1, 0.1, 10, 3, 0],
        "paramFactors10"       : [1,   1e-3, 1e-3,   1,   1,  1,  1, 1,  1e-6,   1,    1],
        
                              #   N,     t1,   t2, al1, al2,  F,  T, t_trip, SF, offset, brightness)
        "allparamDefvalues"    : [-1,    -1,   -1,  -1,  -1, -1, -1,   -1,   -1,  -1,   -1],
        "paramMinbound"        : [0,   1e-6, 1e-6,   0,   0,  0,  0,    0,    0,-1e6,    0],
        "paramMaxbound"        : [1e6,  1e6,  1e6, 1e6, 1e6,  1,  1,  1e6,  1e6, 1e6,  1e6],
        "fitfunctionName"      : fcs_analytical_2c_anomalous,
        "fitfunctionParamUsed" : [0, 1, 2, 3, 4, 5, 10, 6, 7, 8, 9]
    }
    