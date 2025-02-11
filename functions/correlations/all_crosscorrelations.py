from brighteyes_ffs.tools.calcdist_from_coord import list_of_pixel_pairs_at_distance

def get_params(det_type='Genoa Instruments 5x5'):
    listOfX = []
    listOfY = []
    for vert in range(9):
        for hor in range(9):
            listOfY.append(vert-4)
            listOfX.append(hor-4)
    avList = [list_of_pixel_pairs_at_distance([listOfY[i], listOfX[i]], pixelsOff=[]) for i in range(len(listOfX))]
    avListStr = []
    for avSingleDist in avList:
        avstr = ''
        for j in avSingleDist:
            avstr += str(j[0]) + 'x' + str(j[1]) + '+'
        avListStr.append(avstr[0:-1])
        
    elements = ['V'+str(listOfY[i])+'_H'+str(listOfX[i]) for i in range(len(listOfX))]
    listOfG = ['crossAll']
    mode = 'All cross-correlations'
    shortlabel = 'Crosscorrs'
    average = avListStr
    
    return {
        "mode"       : mode,
        "elements"   : elements,
        "listOfG"    : listOfG,
        "shortlabel" : shortlabel,
        "average" : average,
    }
