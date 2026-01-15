def get_params(det_type='Square 5x5'):
    
    if det_type == 'Square 5x5':
        elements = ['sum1', 'sum5', 'sum9', 'sum13', 'sum21', 'sum25']
        listOfG = ['central', 'C7+11+12+13+17', 'sum3', 'C2+6+7+8+10+11+12+13+14+16+17+18+22', 'C2+6+7+8+10+11+12+13+14+16+17+18+22+1+3+5+9+15+19+21+23', 'sum5']
    elif det_type == 'PDA-23':
        elements = ['central', 'sum7', 'sum19', 'sum23']
        listOfG = ['picentral', 'piring1', 'piring2', 'piring3']
    else:
        # airyscan
        elements = ['central', 'ring1', 'ring2', 'ring3']
        listOfG = ['x0000', 'C0+1+2+3+4+5+6',  "C0" + "".join([f"+{i}" for i in range(1, 19)]), "C0" + "".join([f"+{i}" for i in range(1, 32)])]
    
    return {
        "mode"       : 'Spot-variation fcs - extended',
        "elements"   : elements,
        "listOfG"    : listOfG,
        "shortlabel" : 'SpotVarExt',
    }
