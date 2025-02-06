def get_params(det_type='Square 5x5'):
    
    if det_type == 'Square 5x5':
        elements = ['s0', 's1', 's2', 's4', 's5', 's8']
        listOfG = ['crossAll']
        average = ['12x12', '12x7+12x11+12x13+12x17', '12x6+12x8+12x16+12x18', '12x2+12x10+12x14+12x22', '12x1+12x3+12x5+12x9+12x15+12x19+12x21+12x23', '12x0+12x4+12x20+12x24']
    
    else:
        # airyscan
        elements = ['s1', 's2', 's3', 's4']
        listOfG = ['crossAll']
        average = ['0x1+0x2+0x3+0x4+0x5+0x6', '0x7+0x9+0x11+0x13+0x15+0x17', '0x8+0x10+0x12+0x14+0x16+0x18', '0x23+0x24+0x25+0x26+0x27+0x28+0x29+0x30+0x19+0x20+0x21+0x22']
    
    return {
        "mode"       : 'Pair-correlation fcs',
        "elements"   : elements,
        "listOfG"    : listOfG,
        "shortlabel" : 'PairCorr',
        "average" : average,
    }
