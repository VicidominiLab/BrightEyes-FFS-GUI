def get_params(det_type='Square 5x5'):
    
    if det_type == 'Square 5x5':
        elements = ['Right', 'Up', 'Left', 'Down']
        listOfG = ['x1213', 'x1207', 'x1211', 'x1217']
    
    elif det_type == 'PDA-23':
        elements = ['Right', 'UpRight', 'UpLeft', 'Left', 'DownLeft', 'DownRight']
        listOfG = ['x2019', 'x2024', 'x2025', 'x2021', 'x2016', 'x2015']
    
    elif det_type == 'Airyscan 32':
        # airyscan
        elements = ['Right', 'UpRight', 'UpLeft', 'Left', 'DownLeft', 'DownRight']
        listOfG = ['x0015', 'x0028', 'x0007', 'x0009', 'x0011', 'x0013']
    
    else:
        # PRISM
        elements = ['Right', 'Up', 'Left', 'Down']
        listOfG = ['x2425', 'x2417', 'x2423', 'x2431']
    
    return {
        "mode"       : 'Asymmetric diffusion analysis',
        "elements"   : elements,
        "listOfG"    : listOfG,
        "shortlabel" : 'AsymmDiff',
    }
