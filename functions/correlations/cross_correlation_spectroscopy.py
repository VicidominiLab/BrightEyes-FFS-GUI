def get_params(det_type='Square 5x5'):
    elements = ['cross12', 'cross21', 'auto1', 'auto2']
    listOfG = ['2MPD']
    mode = 'Two channel cross-correlation'
    shortlabel = 'CrossCorr'
    
    return {
        "mode"       : mode,
        "elements"   : elements,
        "listOfG"    : listOfG,
        "shortlabel" : shortlabel,
    }
