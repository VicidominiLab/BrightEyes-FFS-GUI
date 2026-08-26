def get_params(det_type='Square 5x5'):
    
    if det_type == 'Square 5x5':
        elements = ['sum1', 'sum5', 'sum9', 'sum13', 'sum21', 'sum25']
        listOfG = ['central', 'C7+11+12+13+17', 'sum3', 'C2+6+7+8+10+11+12+13+14+16+17+18+22', 'C2+6+7+8+10+11+12+13+14+16+17+18+22+1+3+5+9+15+19+21+23', 'sum5']
    elif det_type == 'PDA-23':
        elements = ['central', 'sum7', 'sum19', 'sum23']
        listOfG = ['picentral', 'piring1', 'piring2', 'piring3']
    elif det_type == 'Airyscan 32':
        # airyscan
        elements = ['central', 'ring1', 'ring2', 'ring3']
        listOfG = ['x0000', 'C0+1+2+3+4+5+6',  "C0" + "".join([f"+{i}" for i in range(1, 19)]), "C0" + "".join([f"+{i}" for i in range(1, 32)])]
    else:
        elements = ['central', 'sum5', 'sum9', 'sum13', 'sum21', 'sum37', 'sum49']
        listOfG = ['prismcentral', 'C17+23+24+25+31', 'prismsum3',
                   'C10+16+17+18+22+23+24+25+26+30+31+32+38',
                   'C10+16+17+18+22+23+24+25+26+30+31+32+38+9+11+15+19+29+33+37+39',
                   'C10+16+17+18+22+23+24+25+26+30+31+32+38+9+11+15+19+29+33+37+39+2+3+4+8+12+14+20+21+27+28+38+36+40+44+45+46',
                   "prismsum7"]
        
    return {
        "mode"       : 'Spot-variation fcs - extended',
        "elements"   : elements,
        "listOfG"    : listOfG,
        "shortlabel" : 'SpotVarExt',
    }
