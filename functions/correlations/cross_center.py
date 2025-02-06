def get_params(det_type='Square 5x5'):
    
    if det_type == 'Square 5x5':
        elements = ['Right', 'Up', 'Left', 'Down']
        listOfG = ['crossAll']
        average = ['7x8+12x13+17x18+6x7+11x12+16x17', '12x7+11x6+13x8+17x12+16x11+18x13', '7x6+12x11+17x16+8x7+13x12+18x17', '11x16+12x17+13x18+6x11+7x12+8x13']
    else:
        # airyscan
        elements = ['UpRight', 'UpLeft', 'DownLeft', 'DownRight']
        listOfG = ['crossAll']
        average = ['14x12+15x3+4x11+16x0+5x2+0x10+17x1+6x9+18x8', '10x12+9x3+2x13+8x0+1x4+0x14+7x5+6x15+18x16', '12x14+3x15+11x4+0x16+2x5+10x0+1x17+9x6+8x18', '12x10+3x9+13x2+0x8+4x1+14x0+5x7+15x6+16x18']
    
    return {
        "mode"       : 'Cross-correlation for flow analysis',
        "elements"   : elements,
        "listOfG"    : listOfG,
        "shortlabel" : 'XCorrCenter',
        "average" : average,
    }
