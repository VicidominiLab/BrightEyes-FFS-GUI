# the custom function is needed in case the user adds a .csv file with the correlation
def get_params(det_type='Genoa Instruments 5x5'):
    return {
        "mode"       : 'Custom',
        "elements"   : ['det0_averageX'],
        "listOfG"    : ['det0_averageX'],
        "shortlabel" : 'custom_corr',
    }