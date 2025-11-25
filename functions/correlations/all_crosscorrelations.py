from brighteyes_ffs.tools.calcdist_from_coord import list_of_pixel_pairs_at_distance

def get_params(det_type='Square 5x5'):
    if det_type=='Square 5x5':
        listOfX = []
        listOfY = []
        for vert in range(9):
            for hor in range(9):
                listOfY.append(vert-4)
                listOfX.append(hor-4)
        avList = [list_of_pixel_pairs_at_distance([listOfY[i], listOfX[i]], pixelsOff=[0, 1, 3, 4, 5, 9, 15, 19, 20, 21, 23, 24]) for i in range(len(listOfX))]
        idx_empty = [i for i in range(len(avList)) if avList[i] == []]
        avList = [avList[i] for i in range(len(avList)) if i not in idx_empty]
        listOfX = [listOfX[i] for i in range(len(listOfX)) if i not in idx_empty]
        listOfY = [listOfY[i] for i in range(len(listOfY)) if i not in idx_empty]
        
        avListStr = []
        for avSingleDist in avList:
            avstr = ''
            for j in avSingleDist:
                avstr += str(j[0]) + 'x' + str(j[1]) + '+'
            avListStr.append(avstr[0:-1])
            
        elements = ['V'+str(listOfY[i])+'_H'+str(listOfX[i]) for i in range(len(listOfX))]
        average = avListStr
    
    elif det_type == 'PDA-23':
        elements = []
        for angle in [0, 60, 120, 180, 240, 300]:
            for dist in [1, 2]:
                elements.append('Angle' + str(angle) + '_' + str(dist))
        average = ['12x11+11x10+17x16+16x15+15x14+22x21+21x20+20x19+19x18+26x25+25x24+24x23+30x29+29x28',
                   '12x10+17x15+16x14+22x20+21x19+20x18+26x24+25x23+30x28',
                   '10x14+14x18+11x15+15x19+19x23+12x16+16x20+20x24+24x28+17x21+21x25+25x29+22x26+26x30',
                   '10x18+11x19+15x23+12x20+16x24+20x28+17x25+21x29+22x30',
                   '12x17+17x22+11x16+16x21+21x26+10x15+15x20+20x25+25x30+14x19+19x24+24x29+18x23+23x28',
                   '12x22+11x21+16x26+10x20+15x25+20x30+14x24+19x29+18x28',
                   '11x12+10x11+16x17+15x16+14x15+21x22+20x21+19x20+18x19+25x26+24x25+23x24+29x30+28x29',
                   '10x12+15x17+14x16+20x22+19x21+18x20+24x26+23x25+28x30',
                   '14x10+18x14+15x11+19x15+23x19+16x12+20x16+24x20+28x24+21x17+25x21+29x25+26x22+30x26',
                   '18x10+19x11+23x15+20x12+24x16+28x20+25x17+29x21+30x22',
                   '17x12+22x17+16x11+21x16+26x21+15x10+20x15+25x20+30x25+19x14+24x19+29x24+23x18+28x23',
                   '22x12+21x11+26x16+20x10+25x15+30x20+24x14+29x19+28x18'
                   ]
    
    else:
        elements = ['Angle0', 'Angle60', 'Angle120', 'Angle180', 'Angle240', 'Angle300']
        average = ['0x18+3x6+12x0+2x7+11x1+10x8+4x17+13x5+14x16',
                   '0x16+2x5+10x0+1x17+9x6+8x18+3x15+11x4+12x14',
                   '0x14+1x4+8x0+2x13+9x3+10x12+6x15+7x5+18x16',
                   '0x12+6x3+18x0+17x4+5x13+16x14+7x2+1x11+8x10',
                   '0x10+5x2+16x0+6x9+17x1+18x8+15x3+4x11+14x12',
                   '0x8+4x1+14x0+3x9+13x2+12x10+5x7+15x6+16x18']
    
    listOfG = ['crossAll']
    mode = 'All cross-correlations'
    shortlabel = 'Crosscorrs'
    
    return {
        "mode"       : mode,
        "elements"   : elements,
        "listOfG"    : listOfG,
        "shortlabel" : shortlabel,
        "average" : average,
    }
