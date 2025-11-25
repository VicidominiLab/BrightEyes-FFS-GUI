import ast
from brighteyes_ffs.fcs_gui.restore_session import read_text

# custom function

def get_params(det_type='Genoa Instruments 5x5'):
    try:
        file_txt = read_text('files/custom_functions/custom_corr.txt')
        lines = file_txt.split("\n")
        listOfG = ast.literal_eval(lines[0])  # which correlations to calculate
        elements = ast.literal_eval(lines[1])
        average = ast.literal_eval(lines[2])
        return {
            "mode"       : 'Custom',
            "elements"   : elements,
            "listOfG"    : listOfG,
            "shortlabel" : 'custom_corr',
            "average" :    average,
        }
    except:
        return {
            "mode"       : 'Custom',
            "elements"   : ['det0_averageX'],
            "listOfG"    : ['det0_averageX'],
            "shortlabel" : 'custom_corr',
        }