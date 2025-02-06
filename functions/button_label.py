import os
import sys

current = os.path.dirname(os.path.realpath('button_label.py'))
parent = os.path.dirname(current)
sys.path.append(parent)

from .appearance import corr_short_label

def button_label(file):
    # file is FFSfile object
    coords = file.metadata.coords
    # update label
    labelTitle = file.label
    if labelTitle is None:
        labelTitle = "y = " + str(coords[0]) + ", x = " + str(coords[1])
    # add jobs to label
    corrjobs = ""
    analysisList = file.analysis_list
    for j in range(len(analysisList)):
        analysis = analysisList[j]
        corrlabel = corr_short_label(analysis.mode)
        corrlabel += " - " + str(analysis.settings.resolution)
        corrlabel += "/" + str(analysis.settings.chunksize)
        corrjobs += "\n" + corrlabel
    return [labelTitle, corrjobs]

def analyis_menu_labels(fileAnalysisList):
    # file is FFSsettings.analysisList object
    anMenu = []
    i = 0
    for analysis in fileAnalysisList:
        anMenu.append(str(i+1) + ") " + analysis.mode + " " + str(analysis.settings.resolution) + "/" + str(analysis.settings.chunksize))
        i += 1
    return anMenu

def fit_menu_labels(fileFitList):
    # file is FFSsettings.analysisList object
    fitMenu = []
    i = 0
    for fit in fileFitList:
        fitMenu.append(str(i+1) + ") " + fit.Central.fitfunctionLabel)
        i += 1
    return fitMenu