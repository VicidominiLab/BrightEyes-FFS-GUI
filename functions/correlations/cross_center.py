def get_params(det_type='Square 5x5'):
    
    if det_type == 'Square 5x5':
        elements = ['Right', 'Up', 'Left', 'Down']
        listOfG = ['crossAll']
        average = ['7x8+12x13+17x18+6x7+11x12+16x17',
                   '12x7+11x6+13x8+17x12+16x11+18x13',
                   '7x6+12x11+17x16+8x7+13x12+18x17',
                   '11x16+12x17+13x18+6x11+7x12+8x13']
    elif det_type == 'PDA-23':
        # luminosa
        listOfG = ['crossAll']
        elements = ['Angle0', 'Angle60', 'Angle120', 'Angle180', 'Angle240', 'Angle300']
        average = ['12x10+17x15+16x14+22x20+21x19+20x18+26x24+25x23+30x28',
                   '10x18+11x19+15x23+12x20+16x24+20x28+17x25+21x29+22x30',
                   '12x22+11x21+16x26+10x20+15x25+20x30+14x24+19x29+18x28',
                   '10x12+15x17+14x16+20x22+19x21+18x20+24x26+23x25+28x30',
                   '18x10+19x11+23x15+20x12+24x16+28x20+25x17+29x21+30x22',
                   '22x12+21x11+26x16+20x10+25x15+30x20+24x14+29x19+28x18']
    else:
        # airyscan
        listOfG = ['crossAll']
        elements = ['Angle0', 'Angle60', 'Angle120', 'Angle180', 'Angle240', 'Angle300']
        average = ['0x18+3x6+12x0+2x7+11x1+10x8+4x17+13x5+14x16',
                   '0x16+2x5+10x0+1x17+9x6+8x18+3x15+11x4+12x14',
                   '0x14+1x4+8x0+2x13+9x3+10x12+6x15+7x5+18x16',
                   '0x12+6x3+18x0+17x4+5x13+16x14+7x2+1x11+8x10',
                   '0x10+5x2+16x0+6x9+17x1+18x8+15x3+4x11+14x12',
                   '0x8+4x1+14x0+3x9+13x2+12x10+5x7+15x6+16x18']
    
    return {
        "mode"       : 'Cross-correlation for flow analysis',
        "elements"   : elements,
        "listOfG"    : listOfG,
        "shortlabel" : 'XCorrCenter',
        "average" : average,
    }
