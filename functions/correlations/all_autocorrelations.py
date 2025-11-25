def get_params(det_type='Square 5x5'):
    N = 25 # default
    
    if det_type in ['Square 5x5', 'Genoa Instruments 5x5', 'Nikon NSPARC']:
        N = 25
        listOfG = [i for i in range(N)]
        elements = ['det' + str(i) + 'x' + str(i) for i in range(N)]
    
    elif det_type == 'Genoa Instruments PRISM 7x7':
        N = 49
        listOfG = [i for i in range(N)]
        elements = ['det' + str(i) + 'x' + str(i) for i in range(N)]
    
    elif det_type == 'PDA-23':
        N = 23
        listOfG = [i+9 for i in range(N)]
        elements = ['det' + str(i+9) + 'x' + str(i+9) for i in range(N)]
    
    elif det_type == 'Airyscan 32':
        N = 32
        listOfG = [i for i in range(N)]
        elements = ['det' + str(i) + 'x' + str(i) for i in range(N)]
    
    mode = 'All autocorrelations'
    shortlabel = 'Autocorrs'
    
    return {
        "mode"       : mode,
        "elements"   : elements,
        "listOfG"    : listOfG,
        "shortlabel" : shortlabel,
    }