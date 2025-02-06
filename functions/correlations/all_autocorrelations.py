def get_params(det_type='Square 5x5'):
    N = 25 # default
    if det_type in ['Square 5x5', 'Genoa Instruments 5x5', 'Nikon NSPARC']:
        N = 25
    elif det_type == 'Genoa Instruments PRISM 7x7':
        N = 49
    elif det_type == 'PI Imaging SPAD23':
        N = 23
    elif det_type == 'Airyscan 32':
        N = 32
        
    elements = ['det' + str(i) + 'x' + str(i) for i in range(N)]
    listOfG = [i for i in range(N)] # self.listOfG = ['crossAll']
    mode = 'All autocorrelations'
    shortlabel = 'Autocorrs'
    
    return {
        "mode"       : mode,
        "elements"   : elements,
        "listOfG"    : listOfG,
        "shortlabel" : shortlabel,
    }