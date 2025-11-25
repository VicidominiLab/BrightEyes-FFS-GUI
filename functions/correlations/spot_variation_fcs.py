def get_params(det_type='Square 5x5'):
    # detType = ['Genoa Instruments 5x5', 'Genoa Instruments PRISM 7x7', 'Nikon NSPARC', 'PI Imaging SPAD23']
    if det_type == 'Square 5x5':
        elements = ['central', 'sum3', 'sum5']
        listOfG = ['central', 'sum3', 'sum5']
    elif det_type == 'PDA-23':
        elements = ['central', 'sum7', 'sum19']
        listOfG = ['x2020', 'C15+16+19+20+21+24+25', 'C10+11+12+14+15+16+17+18+19+20+21+22+23+24+25+26+28+29+30']
    else:
        # airyscan
        elements = ['central', 'ring1', 'ring2', 'ring3']
        listOfG = ['x0000', 'C0+1+2+3+4+5+6',  "C0" + "".join([f"+{i}" for i in range(1, 19)]), "C0" + "".join([f"+{i}" for i in range(1, 32)])]
    
    return {
        "mode"       : 'Spot-variation fcs',
        "elements"   : elements,
        "listOfG"    : listOfG,
        "shortlabel" : 'SpotVar',
    }
