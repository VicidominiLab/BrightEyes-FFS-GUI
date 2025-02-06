from brighteyes_ffs.tools.calcdist_from_coord import list_of_pixel_pairs_at_distance

def get_params(det_type='Genoa Instruments 5x5'):
    listOfX = []
    listOfY = []
    for vert in range(5):
        for hor in range(5):
            listOfY.append(vert-2)
            listOfX.append(hor-2)
    avList = [list_of_pixel_pairs_at_distance([listOfY[i], listOfX[i]], pixelsOff=[0,1,3,4,5,9,15,19,20,21,23,24]) for i in range(25)]
    avListStr = []
    for avSingleDist in avList:
        avstr = ''
        for j in avSingleDist:
            avstr += str(j[0]) + 'x' + str(j[1]) + '+'
        avListStr.append(avstr[0:-1])
        
    elements = ['V'+str(listOfY[i])+'_H'+str(listOfX[i]) for i in range(25)]
    listOfG = ['crossAll']
    mode = 'All cross-correlations'
    shortlabel = 'Autocorrs'
    average = avListStr
    
    return {
        "mode"       : mode,
        "elements"   : elements,
        "listOfG"    : listOfG,
        "shortlabel" : shortlabel,
        "average" : average,
    }
