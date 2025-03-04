"""
TO DO:
    URGENT
    ---------------------------------------------------------------------
    software may crash when fit residuals not finite in initial point
        
    CALCULATING CORRELATIONS
    ---------------------------------------------------------------------
    check h5 file
    photobleaching correction https://lsa.umich.edu/content/dam/biophysics-assets/biophysics-SMARTdocs/Hodges_et_al-2018-Journal_of_Fluorescence.pdf
    parallel computing fcs curve
    corr based on macrotime ttm data
    
    FITTING CORRELATIONS
    ---------------------------------------------------------------------
    Bayesian information criterion
    standard error of correlation points, weighted fits
    add all fit functions from pycorrfit
    add fit standard error, reduced chi squared
    shg model
    sted-fcs model
    handle unknown fit model when loading ffs file
    unfittable parameters
    
    OUTPUT
    ---------------------------------------------------------------------
    save figures to file
    save fit results to file
    save active image and active file
    save session does not work when window opened and then cancelled
    
    OTHER
    ---------------------------------------------------------------------
    new session: update folder and file name
    Add date to saved object
    new session returns error if no fit
    convert amplitude to N
    check imFCS plugin imageJ
    (check normalization multipletau)
     --> add option to read arrival times
    load metadate from h5 file (including coordinates)
    color bar finger print
    change file name
    # chunks on > 0
    phasor plot$
    MSD analysis
    all correlations current file does not work
    set chunknr to zero after calc corr
    set chunknr to zero when changing analysis
    set imagenr to zero when session is loaded
    user help
    fitrange update
    https://pyinstaller.readthedocs.io/en/stable/usage.html
 
    
+------------------------- to generate an .exe -------------------------------+
    open anaconda prompt as admin
    go to ffs_gui folder
    activate ffs_gui environment: conda activate ffs_gui
    execute pyinstaller main.spec
    
    pip install "pydantic<2.0"
+-----------------------------------------------------------------------------+
    
    old version:
    execute pyinstaller.exe --onefile --icon=files\facts_icon.png --windowed --paths=..\fcs_gui main.py
    make sure .ui is converted to .py: pyuic5 brighteyes_ffs_3.ui -o pyqt_gui_brighteyes_ffs.py
                                        pyuic5 -x AnalysisDesign.ui -o AnalysisDesign.py
                                        
          
    dependencies:
        numpy
        pyqt5
        matplotlib
        pyinstaller
        h5py
        multipletau
        tifffile
        seaborn -> requires pandas, scipy??
        if analysis_settings not found:
            import sys
            sys.path.insert(0,'/path/to/mod_directory')
            
        opencv-python
        imutils
        scikit-image
    
"""

from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeWidgetItem, QMessageBox, QSplashScreen
from PyQt5.uic import loadUi
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtTest
import numpy as np
import os
import sys
import threading
import matplotlib.patches as patches
import time
import datetime
import subprocess

import qdarkstyle

import argparse

from pyqt_gui_brighteyes_ffs import Ui_MainWindow

current = os.path.dirname(os.path.realpath('main.py'))
parent = os.path.dirname(current)
sys.path.append(parent)

from brighteyes_ffs.tools.path2fname import path2fname

from brighteyes_ffs.fcs_gui.analysis_settings import FFSlib, FFSmetadata

from functions.appearance import appearance as ap
from functions.fitmodels import convert_string_2_start_values, list_of_fit_models, get_fit_model_from_name
from functions.correlation_functions import list_of_correlation_functions, get_correlation_object_from_name
from functions.load_ffs_files import open_image, open_image_dialog, open_ffs, open_ffslib, check_file_name
from functions.session_to_notebook import convert_session_to_notebook, plot_session_in_notebook
from functions.button_label import button_label

from brighteyes_ffs.fcs_gui.load_ffs_metadata import load_ffs_metadata
from brighteyes_ffs.fcs_gui.restore_session import savelib, restorelib, save_ffs
from brighteyes_ffs.fcs_gui.timetrace_end import timetrace_end
from brighteyes_ffs.fcs_gui.session_to_excel import lib2excel

from brighteyes_ffs.fcs.fcs2corr import fcs_load_and_corr_split, Correlations
from brighteyes_ffs.fcs.get_det_elem_from_array import get_elsum
from brighteyes_ffs.fcs.fcs_fit import fcs_fit
from brighteyes_ffs.fcs.fcs2difftime import g2difftime
from brighteyes_ffs.fcs.imsd import fcs2imsd
from brighteyes_ffs.fcs.extract_spad_data import extract_spad_data
from brighteyes_ffs.fcs.filter_g import filter_g_obj
from brighteyes_ffs.fcs.plot_airy import plot_fingerprint_airyscan

from brighteyes_ffs.tools.time2string import time2string
from brighteyes_ffs.tools.fit_curve import fit_curve
from brighteyes_ffs.tools.csv2array import csv2array
from brighteyes_ffs.tools.stokes_einstein import stokes_einstein
from brighteyes_ffs.tools.color_from_map import color_from_map
from brighteyes_ffs.tools.cmaps import change_color_from_map

from brighteyes_ffs.fcs_gui.analysis_settings import FFSfile

def default_settings(self):
    self.ffslib = FFSlib() # main object containing ALL experimental (meta)data
    mdObject = FFSmetadata(time_resolution=10, duration=150, coords=[0, 0]) # default metadata object in case no metadata is found
    self.defaultMetaData = mdObject
    self.firstFile = 0 # first file in the file list shown on top
    self.activeFileButton = 0 # number between 0-5
    #self.activeImage = 0
    airyDummy = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 4], [3, 4, 5, 4, 3], [4, 5, 4, 3, 2], [5, 4, 3, 2, 1]])
    self.w0 = [180, 220, 260] # nm
    self.airyDummy = airyDummy
    self.xtrace = np.arange(0, 100, 1) # default time trace
    self.ytrace = np.zeros((100, 1))
    self.G = None # store calculated G temporally here (needed for threading)
    self.data = None # store calculated time trace temporally here (needed for threading)
    self.finished = False
    self.finishedG = False
    self.updatechunks = True
    self.progress = 0
    self.progressMessage = ''
    self.altFolderPath = '' # alternative folder path to look for files
    self.filePath = '' # path (including file name) to the .ffs file to store sessions
    self.FCSFolderName = 'C:\\Users\\Me\\My fcs folder'
    self.FCSFileName = 'My fcs file'
    self.ui.fitModel_dropdown.clear()
    self.ui.fitModel_dropdown.addItems(list_of_fit_models())
    self.ui.corrs_dropdown.clear()
    self.ui.corrs_dropdown.addItems(list_of_correlation_functions())

def update_plots(self, updateAll=[True, True, True, True], updatechunks=True):
    print('update plots')
    # updateAll boolean array [updateImage, update_timetrace, updateFP, update_analysis]
    file = getfile(self)
    if updatechunks:
        print('update update_chunks')
        update_chunks(self)
        
    if updateAll[0]:
        print('update plot_image')
        plot_image(self)
        
    if updateAll[1]:
        print('update update_timetrace')
        update_timetrace(self, file)
        
    if updateAll[2]:
        print('update update_fingerprint')
        update_fingerprint(self, file)
        
    if updateAll[3]:
        print('update update_analysis')
        update_analysis(self, file)
        #update_chunks(self)
        plot_difflaw(self)

def change_image(self):
    # user moved to next or previous image, now update buttons accordingly
    update_active_button(self)
    update_chunks(self)
    update_buttons(self)
    update_plots(self)

def plot_image(self):
    print('plot image')
    imageNr = self.ffslib.active_image
    imageName = self.ffslib.get_image_name(imageNr)
    imageName = os.path.basename(imageName)
    Nimages = self.ffslib.num_images
    if Nimages == 0:
        imageNr = -1
    image = self.ffslib.return_image(imageNr)
    self.ui.image_widget.canvas.axes.clear()
    self.ui.image_widget.canvas.axes.set_facecolor((0, 0, 0))
    self.ui.image_widget.canvas.axes.set_axis_off()
    self.ui.image_widget.canvas.axes.imshow(image, cmap='hot')
    imageObj = self.ffslib.get_image(imageNr)
    if imageObj is not None:
        activeFile = imageObj.active_ffs
        for f in range(imageObj.num_files):
            if f == activeFile:
                rcolor = "white"
                lw = 1.2
            else:
                rcolor = "yellow"
                lw = 0.7
            # plot circles for ffs positions
            imageShape = np.shape(image)
            file = imageObj.get_ffs_file(f)
            coords = file.coords
            if coords is not None and coords[0] > 0 and coords[0] < imageShape[0] and coords[1] > 0 and coords[1] < imageShape[1]:
                # draw circle
                phi = np.linspace(0, 360)
                r = np.round(imageShape[0] / 70)
                x = coords[1] + r * np.cos(phi * np.pi / 180)
                y = coords[0] + r * np.sin(phi * np.pi / 180)
                self.ui.image_widget.canvas.axes.plot(x, y, linewidth=lw, color=rcolor)
    self.ui.image_widget.canvas.draw()
    self.ui.imageName_button.setText(imageName[0:40] + '\n' + str(imageNr+1) + '/' + str(Nimages))
    
    
    if imageObj is not None:
        mdata = imageObj.print_image_metadata()
        self.ui.imageInfo_label.setText(mdata)

def plot_difflaw(self):
    print('plotdifflaw')
    self.ui.difflaw_widget.canvas.axes.clear()
    self.ui.difflaw_widget.canvas.axes.set_aspect('auto')
    self.ui.difflaw_widget.canvas.draw()
    analysis = getanalysis(self)
    if analysis is not None:
        fit = analysis.return_fit_obj()
        if fit is not None:
            modelname = fit.fit_all_curves[0].fitfunction_label
            model = get_fit_model_from_name(modelname)
            if model is None:
                return
            if model.model not in ['Maximum entropy method free diffusion', 'Flow heat map', 'Model-free displacement analysis', 'Mean squared displacement', 'Asymmetry heat map']:
                param = model.param_names
                
                tau_found, N_found, SP_found = True, True, True
                
                # find index of tau
                ind = [idx for idx in range(len(param)) if 'Tau' in param[idx]]
                if len(ind) < 1:
                    tau_found = False
                else:
                    idxtau = ind[0]
                
                # find index of N
                ind = [idx for idx in range(len(param)) if 'N' in param[idx]]
                if len(ind) < 1:
                    N_found = False
                else:
                    idxN = ind[0]
                
                # find index of Shape parameter
                ind = [idx for idx in range(len(param)) if 'Shape parameter' in param[idx]]
                if len(ind) < 1:
                    SP_found = False
                else:
                    idxSP = ind[0]

                allfitresults = fit.fitresults(returntype="array")
                w0 = fit.return_all("w0")
                if w0 is None or None in w0 or 'None' in w0 or 'Non' in self.ui.w0_edit.text():
                    pass
                else:
                    # w0 = convert_string_2_start_values([self.w0_edit.text()], analysis.NcurvesMode)
                    # w0 = w0[0,:]
                    w0 = 1e-3 * np.asarray(w0) # µm
                    
                    difflawshow = str(self.ui.difflaw_dropdown.currentText())
                    if difflawshow == "Diffusion law" and tau_found:
                        # ------------- plot tau vs w2 ------------
                        # fit first order polynomial through data
                        taufit = allfitresults[idxtau, :]
                        fitresult = fit_curve(taufit, w0**2, 'linear', [1, 1], [1, 1], [-1e6, -1e6], [1e6, 1e6], savefig=0)
                        for i in range(len(taufit)):
                            self.ui.difflaw_widget.canvas.axes.scatter(w0[i]**2, taufit[i], color=color_from_map(np.mod(i, 8), startv=0, stopv=8, cmap='Set2'), edgecolors='k', marker='s', s=20, zorder=2)
                        w02fit = np.zeros(len(w0) + 1)
                        w02fit[0] = 0
                        w02fit[1:] = w0**2
                        taufitres = np.zeros(len(w0) + 1)
                        taufitres[0] = fitresult.x[1]
                        taufitres[1:] = taufit - fitresult.fun
                        if fitresult.x[1] < 0:
                            fitlabel = 'y = {A:.2f} x {B:.2f}'.format(A=fitresult.x[0], B=fitresult.x[1])
                        else:
                            fitlabel = 'y = {A:.2f} x + {B:.2f}'.format(A=fitresult.x[0], B=fitresult.x[1])
                        self.ui.difflaw_widget.canvas.axes.plot(w02fit, taufitres, '--', color='k', linewidth=0.7, label=fitlabel, zorder=1)
                        self.ui.difflaw_widget.canvas.axes.legend(fontsize=7, frameon=False)
                        self.ui.difflaw_widget.canvas.axes.set_xscale('linear')
                        self.ui.difflaw_widget.canvas.axes.set_xlim([0, np.max(w0**2)*1.1])
                        self.ui.difflaw_widget.canvas.axes.set_ylim([0, np.max(taufit)*1.1])
                        self.ui.difflaw_widget.canvas.axes.tick_params(axis='both', which='major', labelsize=6)
                        self.ui.difflaw_widget.canvas.axes.set_xlabel('w0^2 (um^2)', fontsize=7)
                        self.ui.difflaw_widget.canvas.axes.set_ylabel('Diffusion time (ms)', fontsize=7)
                        self.ui.difflaw_widget.canvas.figure.tight_layout()
                        self.ui.difflaw_widget.canvas.draw()
                        return
                    elif difflawshow == "Particle number" and N_found and SP_found:
                        # ------------- plot N vs w3 ------------
                        # fit first order polynomial through data
                        Nfit = allfitresults[idxN, :]
                        SPfit = allfitresults[idxSP, :]
                        V = w0**3*SPfit
                        fitresult = fit_curve(Nfit, V, 'linear', [1, 1], [1, 1], [-1e6, -1e6], [1e6, 1e6], savefig=0)
                        for i in range(len(Nfit)):
                            self.ui.difflaw_widget.canvas.axes.scatter(V[i], Nfit[i], color=color_from_map(np.mod(i, 8), startv=0, stopv=8, cmap='Set2'), edgecolors='k', marker='s', s=20, zorder=2)
                        Vfit = np.zeros(len(V) + 1)
                        Vfit[0] = 0
                        Vfit[1:] = V
                        Nfitresult = np.zeros(len(w0) + 1)
                        Nfitresult[0] = fitresult.x[1]
                        Nfitresult[1:] = Nfit - fitresult.fun
                        if fitresult.x[1] < 0:
                            fitlabel = 'y = {A:.2f} x {B:.2f}'.format(A=fitresult.x[0], B=fitresult.x[1])
                        else:
                            fitlabel = 'y = {A:.2f} x + {B:.2f}'.format(A=fitresult.x[0], B=fitresult.x[1])
                        self.ui.difflaw_widget.canvas.axes.plot(Vfit, Nfitresult, '--', color='k', linewidth=0.7, label=fitlabel, zorder=1)
                        self.ui.difflaw_widget.canvas.axes.legend(fontsize=7, frameon=False)
                        self.ui.difflaw_widget.canvas.axes.set_xscale('linear')
                        self.ui.difflaw_widget.canvas.axes.set_xlim([0, np.max(Vfit)*1.1])
                        self.ui.difflaw_widget.canvas.axes.set_ylim([0, np.max(Nfit)*1.1])
                        self.ui.difflaw_widget.canvas.axes.tick_params(axis='both', which='major', labelsize=6)
                        self.ui.difflaw_widget.canvas.axes.set_xlabel('V (um^3)', fontsize=7)
                        self.ui.difflaw_widget.canvas.axes.set_ylabel('N', fontsize=7)
                        self.ui.difflaw_widget.canvas.figure.tight_layout()
                        self.ui.difflaw_widget.canvas.draw()
                        return
                    elif difflawshow == 'Number and brightness':
                        print('number and brightness')
                        file = getfile(self)
                        if file is not None:
                            tt = file.timetrace
                            duration = file.duration
                            elements = analysis.settings.list_of_g
                            Nfit = allfitresults[idxN, :]
                            Nchunks = analysis.num_chunks(duration)
                            csize = analysis.settings.chunksize
                            splits = np.arange(0, (Nchunks+1)*csize, csize) # s
                            chunks_off = analysis.settings.chunks_off
                            
                            y = tt[0:timetrace_end(tt),:]
                            x = np.arange(0, duration, )
                            
                            if elements is None or len(elements) != len(Nfit):
                                return
                            
                            pcr = np.zeros(len(elements))
                            for idx_el, element in enumerate(elements):
                                try:
                                    tt_single = extract_spad_data(tt, element)
                                    n_tt = len(tt_single) # number of data points in time trace
                                    n_photons = sum(chunk * np.sum(tt_single[int(start/duration*n_tt):int(end/duration*n_tt)]) for chunk, start, end in zip(chunks_off, splits[:-1], splits[1:]))
                                    tot_time = np.sum(chunks_off) * csize # s
                                except:
                                    tt_single = 0.1
                                    n_photons = 0
                                    tot_time = 1
                                #print('original brightness')
                                #print(str(float(np.sum(tt_single)) / duration / Nfit[idx_el] / 1000))
                                pcr[idx_el] = float(n_photons) / tot_time / Nfit[idx_el] / 1000 # kHz
                                #print(str(float(n_photons) / tot_time / Nfit[idx_el] / 1000))
                                self.ui.difflaw_widget.canvas.axes.scatter(Nfit[idx_el], pcr[idx_el], color=color_from_map(np.mod(idx_el, 8), startv=0, stopv=8, cmap='Set2'), edgecolors='k', marker='s', s=20, zorder=2)
                            self.ui.difflaw_widget.canvas.axes.set_xscale('linear')
                            self.ui.difflaw_widget.canvas.axes.set_xlim([0, np.max(Nfit)*1.1])
                            self.ui.difflaw_widget.canvas.axes.set_ylim([0, np.max(pcr)*1.1])
                            self.ui.difflaw_widget.canvas.axes.tick_params(axis='both', which='major', labelsize=6)
                            self.ui.difflaw_widget.canvas.axes.set_xlabel('Number', fontsize=7)
                            self.ui.difflaw_widget.canvas.axes.set_ylabel('Brightness (kHz)', fontsize=7)
                            self.ui.difflaw_widget.canvas.figure.tight_layout()
                            self.ui.difflaw_widget.canvas.draw()
                            return
                           
            
            elif model.model == 'Flow heat map':
                # flow heat map
                try:
                    heatmap, arrow, columnNotFound = fit.fitresults_flowmap()
                except:
                    showdialog('Warning', 'Flow heat map analysis not possible.', 'Calculate all cross-correlations for flow analysis first.')
                    return
                
                if heatmap is None or columnNotFound:
                    showdialog('Warning', 'Flow heat map analysis not possible.', 'Calculate all cross-correlations for flow analysis first.')
                    return
                
                self.ui.difflaw_widget.canvas.axes.imshow(np.flipud(heatmap), cmap='twilight', vmin=-0.7, vmax=0.7)
                phi = np.linspace(0, 2*np.pi, 360)
                R = len(heatmap) / 2
                self.ui.difflaw_widget.canvas.axes.plot(R*np.cos(phi) + R, R*np.sin(phi)+R, '-', color='k', linewidth=5)
                #S = R / 2
                r = arrow[0]
                u = arrow[1]
                self.ui.difflaw_widget.canvas.axes.arrow(90-(r/2), 90-(u/2), r, u, width=1, head_width=4, color='white', length_includes_head=True)
                #self.ui.difflaw_widget.canvas.axes.plot(S*np.cos(phi) + R, S*np.sin(phi)+R, ':', color='w', linewidth=0.7)
                #for phi in [np.pi/4, np.pi/4*3, np.pi/4*5, np.pi/4*7]:
                 #   self.ui.difflaw_widget.canvas.axes.plot([R, R+R*np.cos(phi)], [R,R+ R*np.sin(phi)], ':', color='white')
                self.ui.difflaw_widget.canvas.axes.set_xlim([-0.1*R, 2.1*R])
                self.ui.difflaw_widget.canvas.axes.set_ylim([-0.1*R, 2.1*R])
                self.ui.difflaw_widget.canvas.axes.set_axis_off()
                self.ui.difflaw_widget.canvas.draw()
                return
            
            elif model.model == 'Asymmetry heat map':
                try:
                    heatmap, data_not_found = fit.fitresults_asymmetrymap()
                except:
                    showdialog('Warning', 'Asymmetry map could not be calculated.', 'Perform asymmetric diffusion analysis first (in the correlation panel).')
                    return
                R = len(heatmap) / 2
                phi = np.linspace(0, 2*np.pi, 360)
                self.ui.difflaw_widget.canvas.axes.imshow(np.flipud(heatmap), cmap='jet')
                self.ui.difflaw_widget.canvas.axes.plot(R*np.cos(phi) + R, R*np.sin(phi)+R, '-', color='k', linewidth=5)
                self.ui.difflaw_widget.canvas.axes.set_xlim([-0.1*R, 2.1*R])
                self.ui.difflaw_widget.canvas.axes.set_ylim([-0.1*R, 2.1*R])
                self.ui.difflaw_widget.canvas.axes.set_axis_off()
                self.ui.difflaw_widget.canvas.draw()
                return
            
            elif model.model == 'Model-free displacement analysis':
                try:
                    dtimes, corrv = fit.fitresults_mfda()
                except:
                    showdialog('Warning', 'Model-free displacement analysis not possible.', 'Calculate all cross-correlations to use this analysis.')
                    return
                
                corrv /= np.sum(corrv)
                corrv = (corrv * 125)**2
                xsum = 0
                ysum = 0
                if dtimes is not None:
                    sh = np.shape(dtimes)
                    Nr = sh[0]
                    Nc = sh[1]
                    for i in range(Nr):
                        for j in range(Nc):
                            R = dtimes[i, j]
                            a = i - int(np.floor(Nr/2))
                            b = j - int(np.floor(Nr/2))
                            norm = np.sqrt(a**2 + b**2)
                            add_to_sum = False
                            if norm == 0:
                                norm = 1
                            c = color_from_map(2, startv=0, stopv=8, cmap='Set2')
                            if 0 < i < Nr-1 and 0 < j < Nc-1:
                                c=color_from_map(1, startv=0, stopv=8, cmap='Set2')
                                add_to_sum = True
                            if i==int(np.floor(Nr/2)) and j==int(np.floor(Nc/2)):
                                c=color_from_map(0, startv=0, stopv=8, cmap='Set2')
                            y = -R * a / norm
                            x = R * b / norm
                            if add_to_sum:
                                xsum += x
                                ysum += y
                            self.ui.difflaw_widget.canvas.axes.scatter(1000*x, 1000*y, color=c, marker='o', s=15, edgecolors='k')
                    dt = np.max(1000*dtimes[1:4,1:4])
                    self.ui.difflaw_widget.canvas.axes.arrow(-(1000*xsum/2), -(1000*ysum/2), 1000*xsum, 1000*ysum, width=0.03*dt, head_width=0.09*dt, color=color_from_map(1, startv=0, stopv=8, cmap='Set2'), length_includes_head=True)
                    self.ui.difflaw_widget.canvas.axes.set_xlim([-1.1*dt, 1.1*dt])
                    self.ui.difflaw_widget.canvas.axes.set_ylim([-1.1*dt, 1.1*dt])
                else:
                    showdialog('Warning', 'Not all correlations found.', 'Calculate all cross-correlations to use this analysis.')
                self.ui.difflaw_widget.canvas.axes.set_xlabel('Horizontal diffusion time (ms)', fontsize=7)
                self.ui.difflaw_widget.canvas.axes.set_ylabel('Vertical diffusion time (ms)', fontsize=7)
                self.ui.difflaw_widget.canvas.axes.tick_params(axis='both', which='major', labelsize=6)
                self.ui.difflaw_widget.canvas.draw()
                return
            
            elif model.model == 'Mean squared displacement':
                try:
                    tau, var, varfit = fit.fitresults_msd()
                except:
                    showdialog('Warning', 'Mean squared displacement analysis not possible', 'Calculate all cross-correlations to use this analysis.')
                    return
                
                self.ui.difflaw_widget.canvas.axes.scatter(1e3*tau, var, c='red', marker='o', s=15, edgecolors='k')
                self.ui.difflaw_widget.canvas.axes.plot(1e3*tau, varfit, c='k')
                self.ui.difflaw_widget.canvas.axes.set_xlabel('Time (ms)', fontsize=7)
                self.ui.difflaw_widget.canvas.axes.set_ylabel('Sigma^2', fontsize=7)
                self.ui.difflaw_widget.canvas.axes.tick_params(axis='both', which='major', labelsize=6)
                self.ui.difflaw_widget.canvas.draw()
                return
                
            else:
                # MEM method
                Gsingle = analysis.get_corr()
                try:
                    nparam = 6 if model.shortlabel == 'MEM free diff K' else 5
                    [allfitresults, tauD, _, _] = fit.fitresults_mem(Gsingle[:,0], nparam)
                except:
                    showdialog('Warning', 'MEM analysis not possible', 'Calculate autocorrelations to use this analysis.')
                    return
                
                for i in range(np.shape(allfitresults)[1]):
                    self.ui.difflaw_widget.canvas.axes.scatter(tauD, allfitresults[:,i], color=color_from_map(np.mod(i, 8), startv=0, stopv=8, cmap='Set2'), edgecolors='k', zorder=2, s=6)
                    self.ui.difflaw_widget.canvas.axes.plot(tauD, allfitresults[:,i], color=color_from_map(np.mod(i, 8), startv=0, stopv=8, cmap='Set2'), zorder=3)
                self.ui.difflaw_widget.canvas.axes.set_xlabel('Diffusion time (s)', fontsize=7)
                self.ui.difflaw_widget.canvas.axes.set_ylabel('Relative concentration', fontsize=7)
                self.ui.difflaw_widget.canvas.axes.set_xlim([np.min(tauD), np.max(tauD)])
                self.ui.difflaw_widget.canvas.axes.set_xscale('log')
                self.ui.difflaw_widget.canvas.axes.set_ylim([0, np.max(allfitresults)*1.1])
                self.ui.difflaw_widget.canvas.figure.tight_layout()
                self.ui.difflaw_widget.canvas.draw()
                return

    self.ui.difflaw_widget.canvas.axes.set_xlim([0, 1])
    self.ui.difflaw_widget.canvas.axes.set_ylim([0, 1])
    self.ui.difflaw_widget.canvas.axes.tick_params(axis='both', which='major', labelsize=6)
    self.ui.difflaw_widget.canvas.axes.set_xlabel('w0^2 (um^2)', fontsize=7)
    self.ui.difflaw_widget.canvas.axes.set_ylabel('Diffusion time (ms)', fontsize=7)
    self.ui.difflaw_widget.canvas.figure.tight_layout()                
    self.ui.difflaw_widget.canvas.draw()

def update_timetrace(self, file):
    print('update time trace')
    if file is not None:
        tt = file.timetrace
        duration = file.duration
        N = file.number_of_elements
    else:
        tt = None
        duration = None
    if tt is not None and duration is not None:
        analysis = getanalysis(self)
        if analysis is not None:
            Nchunks = analysis.num_chunks(duration)
            CorrSettings = analysis.settings
            csize = CorrSettings.chunksize
            splits = np.arange(0, (Nchunks+1)*csize, csize)
            chunks_off = CorrSettings.chunks_off
            y = tt[0:timetrace_end(tt),:]
            x = np.arange(0, duration, duration/len(y))
            lenxy = np.min([len(x), len(y)])
            x = x[0:lenxy]
            y = y[0:lenxy]
            yfloat = np.zeros(np.shape(y), dtype=float)
            yfloat[:,:] = y[:,:]
            # yfloat contains time traces of all elements
            yfloat = yfloat / (1000 * duration) * len(y)
            # check which elements have to be plotted
            q = self.ui.showElements_widget.currentItem()
            if q is not None:
                q = q.text()
            else:
                yout = yfloat
            if q == "Central":
                detEl = get_elsum(int(np.sqrt(N)), 0)
                yout = yfloat[:, detEl[0]]
            if q == "Spot-variation":
                # type sum3x3, sum5x5, sum7x7, etc.
                Nrings = int(np.ceil(np.sqrt(N) / 2))
                Nt = int(np.shape(yfloat)[0])
                yout = np.zeros((Nt, Nrings))
                for r in range(Nrings):
                    detEl = get_elsum(int(np.sqrt(N)), r)
                    for det in detEl:
                        yout[:, r] += yfloat[:, det]
            if q == "All individually":
                yout = yfloat
            plot_timetrace(self, x, yout, splits, chunks_off)
        else:
            plot_timetrace(self, self.xtrace, self.ytrace)
    else:
        plot_timetrace(self, self.xtrace, self.ytrace)

def plot_timetrace(self, x, y, splits=None, chunks_off=None):
    print('plot time trace')
    yShape = np.shape(y)
    if len(np.shape(y)) == 1:
        y = np.expand_dims(y, 1)
    yShape = np.shape(y)
    Nplots = yShape[1]
    ymin = np.min(y)
    ymax = np.max(y)
    if ymin == ymax:
        ymax = ymin + 1
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.max((ymin, 0))
    
    self.ui.timetrace_widget.canvas.axes.clear()
    self.ui.timetrace_widget.canvas.axes.set_facecolor((1, 1, 1))
    for i in range(Nplots):
        self.ui.timetrace_widget.canvas.axes.plot(x, y[:,i], color=color_from_map(i, 0, Nplots), linewidth=0.7)
    self.ui.timetrace_widget.canvas.axes.set_xlim([xmin, xmax+x[1]-x[0]])
    self.ui.timetrace_widget.canvas.axes.set_ylim([ymin, ymax])
    self.ui.timetrace_widget.canvas.axes.set_xlabel('Time (s)', fontsize=7)
    self.ui.timetrace_widget.canvas.axes.set_ylabel('Photon flux (kHz)', fontsize=7)
    self.ui.timetrace_widget.canvas.axes.tick_params(axis='both', which='major', labelsize=6)
    
    # plot vertical chunk lines
    if splits is not None:
        if len(splits) <= 101:
            for i in range(len(splits)):
                self.ui.timetrace_widget.canvas.axes.plot([splits[i], splits[i]], [ymin, ymax], color=ap("linecol"), linewidth=0.7)
            # color chunks red if not used for calculating average correlation
            if chunks_off is not None:
                for i in range(len(splits) - 1):
                    edgec = None
                    lw = 0
                    facec = 'white'
                    if not chunks_off[i]:
                        facec = ap("discol")
                    if i == self.ui.chunk_spinBox.value():
                        edgec = ap('chunkbord')
                        lw = 2
                    if not chunks_off[i] or i == self.ui.chunk_spinBox.value():
                        rect = patches.Rectangle((splits[i], ymin), splits[i+1]-splits[i], ymax-ymin, fc=facec, linewidth=lw, edgecolor=edgec)
                        self.ui.timetrace_widget.canvas.axes.add_patch(rect)
        else:
            i = self.ui.chunk_spinBox.value()
            self.ui.timetrace_widget.canvas.axes.plot([splits[i], splits[i]], [ymin, ymax], color=ap('chunkbord'), linewidth=0.7)
    self.ui.timetrace_widget.canvas.draw()

def update_analysis(self, file):
    print('plot corrs')
    # plot correlations
    self.ui.correlations_widget.canvas.axes.clear()
    self.ui.correlations_widget.canvas.axes.set_facecolor((1, 1, 1))
    self.ui.correlations_widget.canvas.axes2.clear()
    self.ui.correlations_widget.canvas.axes2.set_facecolor((1, 1, 1))
    
    plotcolor = 0
    
    if file is not None:
        analysis = file.get_analysis()
        if analysis is not None:
            fits = analysis.return_fit_obj()
            mode = analysis.mode
            x = [0.01, 0.1, 1, 10]
            y = [0, 0, 0, 0]
            xmin = np.min(x)
            xmax = np.max(x)
            ymin = np.min(y)
            ymax = np.max(y)
            yminfit = ymax
            ymaxfit = ymin
            fitfound = False
            Nplots = 0
            
            elements = analysis.settings.elements # central, sum3x3, sum5x5
            corrshow = str(self.ui.showchunkscorr_dropdown.currentText())
            for element in elements:
                if corrshow == "Show average all active chunks" or mode == 'Custom':
                    Gsingle = analysis.get_corr(element)
                else:
                    try:
                        Gsingle = analysis.get_corr(element + '_chunk' + str(self.ui.chunk_spinBox.value()))
                    except:
                        Gsingle = None
                if Gsingle is not None:
                    if mode != "Pair-correlation FCSasdf":
                        x = Gsingle[:, 0]
                        y = np.nan_to_num(Gsingle[:, 1])
                        xmin = np.min((xmin, np.min(x[1:])))
                        xmax = np.max((xmax, np.max(x[1:])))
                        ymin = np.min((ymin, np.min(y[1:])))
                        ymax = np.max((ymax, np.max(y[1:])))
                        std = 1
                        if np.shape(Gsingle)[1] > 2:
                            std = Gsingle[:, 2]
                            #self.correlations_widget.canvas.axes.plot(x[1:], y[1:]+std[1:])
                            #self.correlations_widget.canvas.axes.plot(x[1:], y[1:]-std[1:])
                        self.ui.correlations_widget.canvas.axes.scatter(x[1:], y[1:], s=3, label=element, color=color_from_map(np.mod(plotcolor, 8), startv=0, stopv=8, cmap='Set2'))
                        plotcolor += 1
                        Nplots += 1
                    # elif mode == "Pair-correlation fcs":
                    #     # pair correlation fcs
                    #     x = analysis.getCorr()[:,0]
                    #     xmin = np.min((xmin, np.min(x[1:])))
                    #     xmax = np.max((xmax, np.max(x[1:])))
                    #     crossLabels = ['s = 0', 's = 1', 's = sqrt(2)', 's = 2', 's = sqrt(5)', 's = 2*sqrt(2)']
                    #     for j in range(np.shape(Gsingle)[1]):
                    #         y = Gsingle[:, j]
                    #         ymin = np.min((ymin, np.min(y[1:])))
                    #         ymax = np.max((ymax, np.max(y[1:])))
                    #         self.correlations_widget.canvas.axes.scatter(x[1:], y[1:], s=5, label=crossLabels[j])
                    #         Nplots += 1
                    
                    # check for fit results
                    if fits is not None and corrshow == "Show average all active chunks":
                        fit = fits.fit_all_curves # fit contains list of the 3 spotvar fits
                        if fit[0].fitfunction_label not in ['Model-free displacement analysis', 'Mean squared displacement']:
                            for j in range(len(fit)):
                                if element == fit[j].data:
                                    # fit found
                                    fitfound = True
                                    fitres = fit[j].fitresult
                                    fitrange = fit[j].fitrange
                                    start = fitrange[0]
                                    stop = fitrange[1]
                                    yminfit = np.min((yminfit, np.min(y[start:stop])))
                                    ymaxfit = np.max((ymaxfit, np.max(y[start:stop])))
                                    if fit[0].fitfunction_label not in ['Flow heat map', 'Asymmetry heat map', 'Model-free displacement analysis']:
                                        self.ui.correlations_widget.canvas.axes.plot(x[start:stop], y[start:stop] - fitres, linewidth=0.7, color='black')
                                        self.ui.correlations_widget.canvas.axes2.plot(x[start:stop], fitres, linewidth=0.7, color=color_from_map(np.mod(j, 8), startv=0, stopv=8, cmap='Set2'))
            
            if fitfound:
                xmin = x[start]
                xmax = x[np.min((stop, len(x)))-1]
                ymin = yminfit
                ymax = ymaxfit
        
            self.ui.correlations_widget.canvas.axes.set_xlim([xmin, xmax])
            self.ui.correlations_widget.canvas.axes2.set_xlim([xmin, xmax])
            if Nplots > 0 and Nplots < 13:
                self.ui.correlations_widget.canvas.axes.legend(fontsize=7, frameon=False)
            if ymin is not np.nan and ymax is not np.nan:
                self.ui.correlations_widget.canvas.axes.set_ylim([ymin, np.max((ymin+0.001, ymax))])
    self.ui.correlations_widget.canvas.axes.set_xscale('log')
    self.ui.correlations_widget.canvas.axes.set_ylabel('G', fontsize=7)
    self.ui.correlations_widget.canvas.axes.tick_params(axis='both', which='major', labelsize=6)
    self.ui.correlations_widget.canvas.axes.tick_params(axis='both', which='minor', labelsize=6)
    
    self.ui.correlations_widget.canvas.axes2.set_xscale('log')
    self.ui.correlations_widget.canvas.axes2.set_xlabel('Lag time (s)', fontsize=7)
    self.ui.correlations_widget.canvas.axes2.set_ylabel('Residuals', fontsize=7)
    self.ui.correlations_widget.canvas.axes2.tick_params(axis='both', which='major', labelsize=6)
    self.ui.correlations_widget.canvas.axes2.tick_params(axis='both', which='minor', labelsize=6)
    
    self.ui.correlations_widget.canvas.draw()

def remove_image(self, imageNr='active'):
    currentImage = self.ffslib.active_image
    if currentImage is None:
        return None
    confirm_delete = showdialog('Remove image?', 'Are you sure you want to remove this image? All corresponding correlations and fits will be removed as well. No raw data will be deleted.', '')
    if confirm_delete:
        self.ffslib.remove_image()
        self.activeFileButton = 0
        self.firstFile = 0
        update_buttons(self)
        update_plots(self, updateAll=[True, True, True, True])

def remove_ffs_file(self, fileNr="active"):
    currentImage = self.ffslib.get_image()
    if currentImage is None:
        return None
    # get ffs file for active image
    if fileNr == "active":
        currentImage.remove_ffs_file(self.activeFileButton + self.firstFile)
    else:
        currentImage.remove_ffs_file(fileNr)
    self.activeFileButton = 0
    self.firstFile = 0
    update_buttons(self)
    update_plots(self, updateAll=[True, True, True, True])

def remove_analysis(self, anNr="active"):
    file = getfile(self)
    if file is not None:
        if anNr == "active":
            anNr = file.active_analysis
        file.remove_analysis(anNr)
        update_buttons(self)
        update_plots(self, updateAll=[False, False, False, True])

def remove_fit(self, fitNr="active"):
    analysis = getanalysis(self)
    if analysis is not None:
        if fitNr == "active":
            fitNr = analysis.active_fit
        analysis.remove_fit(fitNr)
        update_buttons(self)
        update_plots(self, updateAll=[False, False, False, True])

def update_fingerprint(self, file):
    fp = None
    duration = None
    if file is not None:
        fp = file.airy
        duration = file.duration
    if fp is not None and duration is not None:
        if len(fp) == 25:
            plot_fingerprint(self, np.reshape(fp, (5,5)) / duration / 1000) # kHz
        if len(fp) == 32:
            # airyscan
            plot_fingerprint(self, fp / duration / 1000)
    else:
        plot_fingerprint(self, self.airyDummy)

def update_chunks(self):
    # user has changed chunk number
    print('update chunks')
    analysis = getanalysis(self)
    file = getfile(self)
    if analysis is not None:
        N = analysis.num_chunks(file.duration)
        c = self.ui.chunk_spinBox.value()
        c = np.clip(c, 0, N-1)
        self.ui.chunk_spinBox.setValue(c)
        chunksOn = analysis.settings.chunks_off
        if chunksOn is not None:
            update_chunk_checkbox(self, bool(chunksOn[c]))

def update_chunk_checkbox(self, chunkBool):
    print('update chunk checkbox')
    self.ui.chunkOn_checkBox.setChecked(chunkBool)

def update_chunks_on(self):
    # user has clicked chunk check box
    print('update chunks on')
    analysis = getanalysis(self)
    if analysis is not None:
        corrsettings = analysis.settings
        chunksOn = corrsettings.chunks_off
        c = self.ui.chunk_spinBox.value()
        chunkOn = self.ui.chunkOn_checkBox.isChecked()
        if np.sum(np.array(chunksOn)) < 2 and chunkOn == False:
            # at least one chunk has to remain on
            QtTest.QTest.qWait(50)
            update_chunk_checkbox(self, True)
            return
        if chunksOn[c] != int(chunkOn):
            chunksOn[c] = int(chunkOn)
            corrsettings.update(chunks_off=chunksOn, analysis=analysis)
            print('redo fit because average has changed')
            perform_fit(self, updateAll=True)
            update_buttons(self)

def update_chunks_on_filter(self):
    # automatic filtering of chunks
    print('filter corr')
    analysis = getanalysis(self)
    if analysis is not None:
        corrsettings = analysis.settings
        ind_on = filter_g_obj(analysis.corrs, filt='sum5', f_acc=0.66)
        if ind_on is None:
            return
        ind_on_original = corrsettings.chunks_off
        if np.array_equal(np.array(ind_on_original), np.array(ind_on)):
            # turn everything on again
            ind_on = np.ones((len(ind_on)))
        corrsettings.update(chunks_off=ind_on, analysis=analysis)
        c = self.ui.chunk_spinBox.value()
        update_chunk_checkbox(self, bool(ind_on[c]))
        perform_fit(self, updateAll=True)
        update_buttons(self)

def plot_fingerprint(self, fp):
    self.ui.fingerprint_widget.canvas.axes.clear()
    self.ui.fingerprint_widget.canvas.axes.set_facecolor((0, 0, 0))
    self.ui.fingerprint_widget.canvas.axes.set_axis_off()
    if len(fp) == 32:
        #ax.set_facecolor("black")
        #self.ui.fingerprint_widget.canvas.axes.set_facecolor(change_color_from_map(ap("subtlecol"))
        s, hexb = plot_fingerprint_airyscan(np.abs(fp), plot=False)
        self.ui.fingerprint_widget.canvas.axes.hexbin(s[1], s[0], C=hexb, gridsize=[6,5], cmap=change_color_from_map('inferno', ap("bgcol")))
        self.ui.fingerprint_widget.canvas.axes.set_xlim([-0.5,5.5])
        self.ui.fingerprint_widget.canvas.axes.set_ylim([2,8.5])
        self.ui.fingerprint_widget.canvas.axes.set_box_aspect(1)
    else:
        self.ui.fingerprint_widget.canvas.axes.imshow(fp, cmap='inferno')
    self.ui.fingerprint_widget.canvas.draw()

def open_ffs_file(self, buttonNr, filepath=None):
    if filepath is None:
        fname = open_ffs()
    else:
        fname = filepath
    currentFileNr = self.firstFile + buttonNr
    Nfiles = nrfiles(self)
    if fname is not None and currentFileNr <= Nfiles:
        # FFS file found --> check for meta data
        try:
            md = load_ffs_metadata(fname)
        except:
            md = self.defaultMetaData
        if md is None:
            md = self.defaultMetaData
        if self.ffslib.num_images < 1:
            # library is empty, add random image, then add ffsfile to image
            self.ffslib.add_random_image()
            self.ffslib.active_image = 0
        currentImage = self.ffslib.get_image()
        # create ffs file object and add to the list
        FFSfileObj = FFSfile()
        FFSfileObj.fname = fname
        FFSfileObj.metadata = md
        [label, dummy] = button_label(FFSfileObj)
        FFSfileObj.label = label
        currentImage.add_ffs_file(FFSfileObj)
        
        if fname[-4:] == '.csv':
            # add correlation
            currentFile = currentImage.get_ffs_file(currentImage.num_files - 1)
            currentFile.add_analysis(mode=get_correlation_object_from_name('Custom'))
            G = Correlations()
            Gtemp = csv2array(fname, dlmt=-1)
            G.det0_chunk0 = Gtemp
            G.det0_averageX = Gtemp
            G.dwellTime = 1e6 * (Gtemp[1,0] - Gtemp[0,0]) # µs
            self.data = np.zeros((100,25))
            analysis = currentFile.get_analysis()
            analysis.update(corrs = G)
            analysis.settings.chunksize = 150
            currentFile.update(timetrace=self.data, airy=self.airyDummy)
        self.activeFileButton = buttonNr
        currentImage.active_ffs = currentFileNr
        #update_buttons(self)
        #update_plots(self)

def importlib(self, fname, dummy=0):
    self.ffslib = restorelib(fname, self)
    self.finishedG = True

def exportlib(self, dummy=0, filename=''):
    fname = savelib(self.ffslib, self, filename)
    self.finishedG = True
    if fname is not None:
        self.filePath = fname

def exportlib_xlsx(self):
    fname = save_ffs(window_title='Export fit results as', ftype='*.xlsx', directory='')
    if fname is None:
        return
    if not fname.endswith('.xlsx'):
        fname = fname + '.xlsx'
    lib2excel(self.ffslib, fname)

def clean_session(self):
    default_settings(self)
    update_buttons(self)
    
def showdialog(title, message, extraInfo):
   msg = QMessageBox()
   msg.setIcon(QMessageBox.Information)
   msg.setText(message)
   msg.setInformativeText(extraInfo)
   msg.setWindowTitle(title)
   #msg.setDetailedText("The details are as follows:")
   msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
   msg.buttonClicked.connect(msgbtn)
   if msg.exec_() == QMessageBox.Ok:
       return True
   return False

def msgbtn(i):
    pass

def open_image_new_thread(self, fname, dummy=0):
    try:
        [image, fname] = open_image(fname, self)
    except:
        image = None
    if image is None:
        self.imagetemp = None
        self.fnametemp = None
    self.imagetemp = image
    self.fnametemp = fname
    self.finishedG = True

def file_buttons(self):
    return [self.ui.FCSfile0_button, self.ui.FCSfile1_button, self.ui.FCSfile2_button, self.ui.FCSfile3_button, self.ui.FCSfile4_button]

def analysis_checkboxes(self):
    return [self.ui.fit0_checkBox, self.ui.fit1_checkBox, self.ui.fit2_checkBox, self.ui.fit3_checkBox, self.ui.fit4_checkBox, self.ui.fit5_checkBox, self.ui.fit6_checkBox, self.ui.fit7_checkBox, self.ui.fit8_checkBox, self.ui.fit9_checkBox, self.ui.fit10_checkBox, self.ui.fitweighted_checkBox]

def analysis_edits(self):
    return [self.ui.fit0_edit, self.ui.fit1_edit, self.ui.fit2_edit, self.ui.fit3_edit, self.ui.fit4_edit, self.ui.fit5_edit, self.ui.fit6_edit, self.ui.fit7_edit, self.ui.fit8_edit, self.ui.fit9_edit, self.ui.fit10_edit]

def getfile(self, mode="active"):
    # return file with number "mode". By default return active file
    # get active image first
    currentImage = self.ffslib.get_image()
    if currentImage is None:
        return None
    # get ffs file for active image
    return currentImage.get_ffs_file(mode)

def getanalysis(self, mode=-1):
    # return analysis object with number "mode". By default return active
    # analysis object
    file = getfile(self)
    if file is None:
        return None
    return file.get_analysis(mode)
    
def getfit(self, mode="active"):
    # return fit object from active analysis object with number "mode".
    # By default return active fit object
    analysis = getanalysis(self)
    if analysis is None:
        return None
    if mode == "active":
        fitNr = analysis.active_fit
    else:
        fitNr = mode
    if fitNr is None or analysis.num_fits < fitNr:
        return None
    return analysis.return_fit_obj(fitNr)

def getanalysis_cboxes(self):
    cboxes = []
    for cbox in analysis_checkboxes(self):
        cboxes.append(cbox.isChecked())
    return cboxes

def getanalysis_edits(self):
    edits = []
    for edit in analysis_edits(self):
        edits.append(edit.text())
    return edits

def nrfiles(self, mode="active"):
    # return the number of FFS files corresponding to a given image
    # By default the active image is assumed
    image = self.ffslib.get_image(mode)
    if image is None:
        return 0
    return image.num_files

def addcorr_to_file(self, fileNr="active"):
    # add correlation analysis to FFS file
    # By default the active FFS file of the active image is assumed
    file = getfile(self, fileNr)
    mode = str(self.ui.corrs_dropdown.currentText()) # string
    det_type = str(self.ui.detector_dropdown.currentText()) # string
    mode = get_correlation_object_from_name(mode, det_type)
    algorithm = str.lower(str(self.ui.algorithm_dropdown.currentText())) # string
    try:
        resolution = int(self.ui.resolution_edit.text())
    except:
        resolution = 10
    try:
        chunksize = float(self.ui.chunkSize_edit.text())
    except:
        chunksize = 10
    if file is not None:
        file.add_analysis(mode, resolution, chunksize, algorithm)
        update_buttons(self)

def nr_corrs(self, fileNr="active"):
    # return the number of correlation analyses for a given FFS files
    # By default the active file of the active image is assumed
    file = getfile(self, fileNr)
    if file is None:
        return 0
    return file.num_analyses

def calc_g_new_thread(self, file, anSettings):
    # calculate G in different thread
    print('start calc G')
    print(file.fname)
        
    if anSettings.list_of_g[0] == "crossAll":
        #check for averaging first
        els = anSettings.elements
        avs = anSettings.average
        if avs is not None:
            averaging = []
            for i in range(len(avs)):
                averaging.append([els[i], avs[i]])
        else:
            averaging = None
        # [G, data] = fcs_sparse_matrices(fname=file.fname,
        #                               accuracy=anSettings.resolution,
        #                               split=anSettings.chunksize,
        #                               time_trace=True,
        #                               return_obj=True,
        #                               averaging=averaging,
        #                               root=self)
        [G, data] = fcs_load_and_corr_split(file.fname,
                                        list_of_g=anSettings.list_of_g,
                                        accuracy=anSettings.resolution,
                                        split=anSettings.chunksize,
                                        time_trace=True,
                                        metadata=file.metadata,
                                        root=self,
                                        averaging=anSettings.average,
                                        list_of_g_out=anSettings.elements,
                                        algorithm="sparse_matrices")
            
    else:
        [G, data] = fcs_load_and_corr_split(file.fname,
                                        list_of_g=anSettings.list_of_g,
                                        accuracy=anSettings.resolution,
                                        split=anSettings.chunksize,
                                        time_trace=True,
                                        metadata=file.metadata,
                                        root=self,
                                        list_of_g_out=anSettings.elements,
                                        algorithm=anSettings.algorithm)
    self.G = G
    self.data = data
    self.finishedG = True

def perform_fit_new_thread(self, tau, G, fitf, farr, startv, minb, maxb, weights):
    try:
        fitresult = fcs_fit(tau, G, fitf, farr, startv, minb, maxb, -1, 0, 0, False, weights)
    except:
        showdialog('Warning', 'Fit unsuccessful.', 'Fit residuals not finite in initial point.')
        fitresult = None
    self.fitresult = fitresult
    self.finishedG = True

def set_fit_modelbox(self, value):
    index = self.ui.fitModel_dropdown.findText(value)
    if index >= 0:
         self.ui.fitModel_dropdown.setCurrentIndex(index)

def update_fit_model(self, values=None, fitbool=None, fitrange=None):
    print('updatefitmodel')
    # values: fitted values for all 11 parameters (7 for circFCS)
    
    acbox = analysis_checkboxes(self)
    aedit = analysis_edits(self)
    Nwidgets = len(acbox) - 1 # last checkbox is weights
    modelname = str(self.ui.fitModel_dropdown.currentText())
    fitmodelList = list_of_fit_models()
    
    if modelname not in fitmodelList:
        modelname = fitmodelList[0]
        values = None
    
    model = get_fit_model_from_name(modelname)
    
    paramNamesAll = ['None' for i in range(Nwidgets)]
    paramFittableAll = [False for i in range(Nwidgets+1)]
    paramDefvaluesAll = ['NaN' for i in range(Nwidgets)]
    fitboolAll = [False for i in range(Nwidgets+1)]
    # get fit model info
    paramNames = model.param_names
    paramFittable = model.param_fittable
    paramDefvalues = np.asarray(model.param_def_values)
    
    # use input values if given
    if values is not None:
        paramDefvalues = values
    paramDefvaluesAll[0:len(paramDefvalues)] = paramDefvalues
    
    if fitbool is not None:
        fitboolAll[0:len(paramNames)] = fitbool[0:len(paramNames)]
        fitboolAll[-1] = fitbool[-1] # check for weighted fit
    paramNamesAll[0:len(paramNames)] = paramNames
    
    if paramFittable is not None:
        paramFittableAll[0:len(paramFittable)] = paramFittable
    
    
    for i in range(Nwidgets):
        acbox[i].setText(paramNamesAll[i])
        acbox[i].setChecked(fitboolAll[i])
        acbox[i].setEnabled(paramFittableAll[i])
        aedit[i].setText(str(paramDefvaluesAll[i]))
        if paramNamesAll[i] == 'None':
            aedit[i].setEnabled(False)
        else:
            aedit[i].setEnabled(True)
            
    acbox[Nwidgets].setText('Weighted fit')
    acbox[Nwidgets].setChecked(fitboolAll[Nwidgets])
    # update fit range
    if fitrange is None:
        fitrange = [1, 1000]
    self.ui.fitstart_spinBox.setValue(fitrange[0])
    self.ui.fitstop_spinBox.setValue(fitrange[1])

def update_difflaw_texts(self):
    an = getanalysis(self)
    if an is not None:
        fit = an.return_fit_obj()
        if fit is not None:
            D = fit.return_field('D')
            w0 = fit.return_field('w0')
            if None in D or None in w0:
                w0 = self.w0
                D = [None]
            # update D
            Dstr = ''.join(str(D[i])[0:4] + ", " for i in range(len(D)))
            Dstr = Dstr[:-2]
            self.ui.D_edit.setText(Dstr) # µm^2/s
            # update w0
            w0str = ''.join(str(w0[i])[0:3] + ", " for i in range(len(w0)))
            w0str = w0str[:-2]
            self.ui.w0_edit.setText(str(w0str)) # nm

def update_fit_analysis(self, ftype='new'):
    analysis = getanalysis(self)
    modelname = str(self.ui.fitModel_dropdown.currentText())
    model = get_fit_model_from_name(modelname)
    if model is not None and analysis is not None and analysis.corrs is not None:
        fitarray = getanalysis_cboxes(self)
        print('update_fit_analysis')
        startvalues = getanalysis_edits(self)
        startvalues = np.asarray(convert_string_2_start_values(startvalues, analysis.n_curves_mode))
        fitrange = [self.ui.fitstart_spinBox.value(), self.ui.fitstop_spinBox.value()]
        if ftype == 'new' or analysis.num_fits < 1:
            analysis.add_fit_analysis(model, fitarray, startvalues, fitrange=fitrange)
        else:
            # update analysis
            analysis.update_fit_analysis(model, fitarray, startvalues, fitrange=fitrange)
        perform_fit(self)
        update_buttons(self)
        

def perform_fit(self, updateAll=False):
    print('perform fit')
    # if updateAll is true, then all fits for the current analysis are redone
    # necessary when a chunk is turned off or on
    
    analysis = getanalysis(self)
    if updateAll:
        fitNr = list(range(analysis.num_fits))
    else:
        fitNr = [-1]
    for fNr in fitNr:
        fitObj = analysis.return_fit_obj(fNr)
        if fitObj is None:
            return
        allfits = fitObj.fit_all_curves
        if allfits is None:
            return
        # perform fit central, sum3, sum5, etc.
        for f in range(len(allfits)):
            # get data
            fit = allfits[f]
            G = analysis.get_corr(fit.data)
            r = fit.fitrange
            fitmodel = get_fit_model_from_name(fit.fitfunction_label)
            if fitmodel is None:
                return
            fitf = fitmodel.fitfunction_name
            farr = fit.fitarray
            startv = fit.startvalues
            
            if 'global fit' in fitmodel.model:
                # break loop of individudal fits and perform global fit
                break
            
            # if fitmodel.model == 'Maximum entropy method free diffusion':
            #     Nvalues = len(startv)
                
            #     if Nvalues > 6:
            #         startvNew = np.zeros((6))
            #         startvNew[0] = Nvalues - 5
            #         startvNew[1:] = startv[-5:]
            #         startv = startvNew
                
            #     # update fit box
                
                
            #     fit = self.fitAllCurves[0]
            #     power10 = fit.paramFactors10
            #     Nparam = 12
            #     stv = ["NaN" for j in range(Nparam)]
            #     for j in range(len(fit.paramidx)):
            #         fitabsv = fit.startvalues[-6+fit.paramidx[j]] / power10[j]
            #         stv[j] = fitabsv
            #     update_fit_model(self, values=stv)
            
            minb = fit.minbound
            maxb = fit.maxbound
            stop = np.min((r[1], len(G)))
            start = np.min((stop - 1, r[0]))
            start = np.max((0, start))
            fit.fitrange = [start, stop]
            weights = 1
            if fitmodel.model not in ['Flow heat map', 'Asymmetry heat map', 'Model-free displacement analysis', 'Mean squared displacement']:
                print('new fit performed')
                if farr[-1] == 1 and np.shape(G)[1] > 2 and np.min(G[:, -1]) > 0:
                    # weighted fit
                    weights = 1 / (G[start:stop, -1]**2) # convert standard deviation to variance
                    weights /= np.min(weights)
                    weights = np.clip(weights, 0, 10) # clip excessive weights
                farr = farr[0:-1]
                if G is not None:
                    startv = np.clip(startv, minb, maxb)
                    self.finishedG = False
                    th = threading.Thread(target=perform_fit_new_thread, args=(self, G[start:stop, 1], G[start:stop, 0], fitf, farr, startv, minb, maxb, weights))
                    # Start the thread
                    th.start()
                    progressTxtStart = "Please wait, performing fit.\nThis may take a while."
                    self.ui.progressBar_label.setText(progressTxtStart)
                    self.update_progress_bar(0)
                    while not self.finishedG:
                        QtTest.QTest.qWait(50)
                        self.ui.progressBar_label.setText(progressTxtStart)
                    th.join()
                    self.update_progress_bar(100)
                    if self.fitresult is not None:
                        fitresult = self.fitresult
                        newv = startv
                        j = 0
                        if fitmodel.model == 'Maximum entropy method free diffusion':
                            if fitmodel.shortlabel == 'MEM free diff K':
                                # new version
                                Ncomp = len(fitresult.x)
                                newv = np.zeros((Ncomp+6))
                                newv[0:Ncomp] = fitresult.x
                                newv[-6:] = startv[-6:] # histogram values
                            else:
                                # old version
                                Ncomp = len(fitresult.x)
                                newv = np.zeros((Ncomp+5))
                                newv[0:Ncomp] = fitresult.x
                                newv[-5:] = startv[-5:] # histogram values
                        else:
                            for i in range(len(farr)):
                                if farr[i]:
                                    newv[i] = fitresult.x[j]
                                    j += 1
                        fit.update(fitresult=fitresult.fun, startvalues=newv)
                   
                    
            elif fitmodel.model == 'Model-free displacement analysis':
                z = []
                difftime, corrv = g2difftime(G[start:stop,:], smoothing=int(startv[0]))
                fit.update(fitresult=np.atleast_1d(np.asarray([difftime, corrv])))
            
            else:
                # flow heat map
                z = []
                try:
                    #z = list(g2polar(G[start:stop, 1]))
                    z = G[start:stop, 1]
                    fit.update(fitresult=z)
                except:
                    pass
        
        if fitmodel.model == 'Mean squared displacement':
            print('Mean squared displacement')
            G3d, tau = analysis.get_corr3D()
            if G3d is None:
                showdialog('Warning', 'Not all correlations found.', 'Calculate all cross-correlations to use MSD analysis.')
                return
            minb = fit.minbound
            maxb = fit.maxbound
            var, tau, fitres = fcs2imsd(G3d[start:stop,:,:], tau[start:stop], farr[0:-1], startv, minb, maxb, remove_outliers=False)
            for f in range(len(allfits)):
                # get data
                fit = allfits[f]
                fit.update(fitresult=np.atleast_1d(np.asarray([tau, var])), startvalues=fitres)
        
        # perform global fit
        if 'global fit' in fitmodel.model:
            tau = G[:, 0]
            Ntraces = len(allfits)
            G = np.zeros((len(G), Ntraces))
            fitInfo = np.zeros((4 + 5*Ntraces, 4)) # fitinfo, startv, minb, maxb
            
            for f in range(Ntraces):
                # get data
                fit = allfits[f]
                Gtemp = analysis.get_corr(fit.data)
                G[:,f] = Gtemp[:,1]
                minb = fit.minbound
                maxb = fit.maxbound
                stop = np.min((r[1], len(G)))
                start = np.min((stop - 1, r[0]))
                start = np.max((0, start))
                fit.fitrange = [start, stop]
                fitarraytemp = fit.fitarray[0:-1]
                fitstartvtemp = fit.startvalues
                
                for idxp, fitp in enumerate([fitarraytemp, fitstartvtemp, minb, maxb]):
                    fitInfo[0:2, idxp] = fitp[0:2] # c, D
                    fitInfo[2+f, idxp] = fitp[2] # w for all
                    fitInfo[2+Ntraces+f, idxp] = fitp[3] # SF for all
                    fitInfo[2+2*Ntraces+f, idxp] = fitp[4] # rhox for all
                    fitInfo[2+3*Ntraces+f, idxp] = fitp[5] # rhoy for all
                    fitInfo[2+4*Ntraces, idxp] = fitp[6] # vx
                    fitInfo[3+4*Ntraces, idxp] = fitp[7] # vy
                    fitInfo[4+4*Ntraces+f, idxp] = fitp[8] # dc for all
                
            # [c, D, w0, ..., wN, SF0, ..., SFN, rhox0, ..., rhoxN, rhoy0, ..., rhoyN, vx, vy, dc0, ..., dcN]
            try:
                fitresult = fcs_fit(G[start:stop,:], tau[start:stop], fitf, fitInfo[:,0], fitInfo[:,1], fitInfo[:,2], fitInfo[:,3], -1, 0, 0, False)
                Ntau = len(tau[start:stop])
                
                fitOut = np.zeros((9, Ntraces))
                for f in range(Ntraces):
                    fitOut[:,f] = allfits[f].startvalues
                j = 0
                if fitarraytemp[0]:
                    fitOut[0,:] = fitresult.x[j] # c was fitted
                    j += 1
                if fitarraytemp[1]:
                    fitOut[1,:] = fitresult.x[j] # D was fitted
                    j += 1
                for p in range(4):
                    if fitarraytemp[p+2]:
                        fitOut[p+2,:] = fitresult.x[j:j+Ntraces] # w, SF, rhox, rhoy were fitted
                        j += Ntraces
                if fitarraytemp[6]:
                    fitOut[6,:] = fitresult.x[j] # vx was fitted
                    j += 1
                if fitarraytemp[7]:
                    fitOut[7,:] = fitresult.x[j] # vy was fitted
                    j += 1
                if fitarraytemp[8]:
                    fitOut[8,:] = fitresult.x[j:j+Ntraces] # dc was fitted
                # beam waist should be the same for all curves
                fitOut[2,:] = fitOut[2,0]
                
                for f in range(Ntraces):
                    fit = allfits[f]
                    fit.update(fitresult=fitresult.fun[f*Ntau:(f+1)*Ntau], startvalues=fitOut[:,f])
            except:
                showdialog('Warning', 'Fit unsuccessful.', 'Fit residuals not finite in initial point.')
                return
                  

def copy_correlation(self):
    file = getfile(self)
    file.copy_correlation(-1)
    update_buttons(self)
    update_plots(self, updateAll=[False, False, False, True])

def use_fit_as_data(self):
    file = getfile(self)
    file.use_fit_as_data(-1,-1)
    update_buttons(self)
    update_plots(self, updateAll=[False, False, False, True])

def update_w0_diff(self):
    fit = getfit(self)
    analysis = getanalysis(self)
    if analysis is not None and fit is not None:
        keep = self.ui.keepFixed_dropdown.currentText()
        w0 = self.ui.w0_edit.text() # nm
        
        D = self.ui.D_edit.text() # µm^2/s
        data = convert_string_2_start_values([w0, D], analysis.n_curves_mode)
        self.w0 = data[0,:]
        # get three fitted tau values
        fitmodelname = fit.fit_all_curves[0].fitfunction_label
        fitmodel = get_fit_model_from_name(fitmodelname)
        if fitmodel is None:
            return
        if fitmodel.model != 'Maximum entropy method free diffusion' and 'global' not in fitmodel.model:
            param = fitmodel.param_names
            ind = [idx for idx in range(len(param)) if 'Tau' in param[idx]]
            if len(ind) != 0:
                idx = ind[0]
                allfitresults = fit.fitresults(returntype="array")
                taufit = allfitresults[idx, :]
                # convert tau to either w0 or D: 4 * D * tau = w0^2
                if keep == "Keep w0 fixed":
                    D = 1e12 * (1e-9*data[0,:])**2 / 4 / (1e-3*taufit) # µm/ 2/s
                    data[1,:] = D
                    Dstr = ''.join(str(D[i])[0:5] + ", " for i in range(len(D)))
                    Dstr = Dstr[:-2]
                    self.ui.D_edit.setText(Dstr)
                else:
                    w0 = 1e9 * np.sqrt(4 * 1e-12*data[1,:] * 1e-3*taufit)
                    data[0,:] = w0
                    w0str = ''.join(str(w0[i])[0:5] + ", " for i in range(len(w0)))
                    w0str = w0str[:-2]
                    self.ui.w0_edit.setText(w0str)
        # update fit objects with w0 and D values
        for i, singlefit in enumerate(fit.fit_all_curves):
            singlefit.update(w0=data[0, i], D=data[1, i])
        plot_difflaw(self)
        
def update_diameter(self):
    fit = getfit(self)
    analysis = getanalysis(self)
    if analysis is not None and fit is not None:
        calc = self.ui.calcDiameter_dropdown.currentText()
        w0 = self.ui.w0_edit.text() # nm
        D = self.ui.D_edit.text() # µm^2/s
        diameter = self.ui.diameter_edit.text() # nm
        visc = self.ui.visc_edit.text() # Pa.s
        T = self.ui.T_edit.text() # K
        
        data = convert_string_2_start_values([w0, D, diameter, visc, T], analysis.n_curves_mode)
        # get three fitted tau values
        fitmodelname = fit.fit_all_curves[0].fitfunction_label
        fitmodel = get_fit_model_from_name(fitmodelname)
        if fitmodel is None:
            return
        if fitmodel.model != 'Maximum entropy method free diffusion':
            #param = fitmodel.paramNames
            #ind = [idx for idx in range(len(param)) if 'Tau' in param[idx]]
            #idx = ind[0]
            #allfitresults = fit.fitresults(returntype="array")
            
            # convert tau to either w0 or D: 4 * D * tau = w0^2
            if calc == "Calculate diameter":     # d = fun(D, T, visc)
                diameter = 1e9 * stokes_einstein(1e-12*data[1,:], data[4,:], data[3,:]) # nm
                data[2,:] = diameter
                diameterstr = ''.join(str(diameter[i])[0:5] + ", " for i in range(len(diameter)))
                diameterstr = diameterstr[:-2]
                self.ui.diameter_edit.setText(diameterstr)
            elif calc == 'Calculate viscosity':
                visc = 1e9 * stokes_einstein(data[1,:], data[4,:], data[2,:]) # Pa.s
                data[3,:] = visc
                viscstr = ''.join(str(visc[i])[0:6] + ", " for i in range(len(visc)))
                viscstr = viscstr[:-2]
                self.ui.visc_edit.setText(viscstr)
            elif calc == 'Calculate temperature':
                T = 1/stokes_einstein((1e-12*data[1,:]), 1/(1e-9*data[2,:]), data[3,:]) # K
                data[4,:] = T
                Tstr = ''.join('{0:.0f}'.format(T[i]) + ", " for i in range(len(T)))
                Tstr = Tstr[:-2]
                self.ui.T_edit.setText(Tstr)
            # update fit objects with diameter values
            #TODO
        
def update_buttons(self, updatechunks=True, updateOnlyButtonName=False):
    print('update buttons')
    self.ui.notes_edit.setPlainText(self.ffslib.notes)
    self.setWindowTitle(f"BrightEyes-FFS - {self.filePath}" if self.filePath else "BrightEyes-FFS")
    Nfiles = nrfiles(self)
    fButtons = file_buttons(self)
    #update figures
    buttonNr = self.activeFileButton
    Newfilebutton = False
    for i in range(5):
        currentFileNr = self.firstFile + i
        if currentFileNr < Nfiles:
            file = getfile(self, currentFileNr)
            fButtons[i].setText(file.label)
            coords = file.metadata.coords
            if i == buttonNr:
                # color this button
                fButtons[i].setStyleSheet("background-color:" + ap("actbut"))
                im = self.ffslib.get_image()
                im.active_ffs = currentFileNr
                
                # update all file settngs
                [filename, folder] = path2fname(file.fname)
                strLen = 80
                if len(filename) > strLen:
                    filename = filename[0:strLen-3] + "..."
                if len(folder) > strLen:
                    folder = folder[0:strLen-3] + "..."
                self.ui.FCSFolderName_label.setText(folder)
                self.ui.FCSFileName_label.setText(filename)
                self.ui.FCSFolderName_label.setStyleSheet("color:" + ap("subtlecol"))
                
                self.ui.label_edit.setText(file.label)
                self.ui.ycoord_edit.setText(str(coords[0]))
                self.ui.xcoord_edit.setText(str(coords[1]))
                
                if updateOnlyButtonName:
                    return
                
                # update analysis menu
                analysisTree = self.ui.correlations_treeWidget
                analysisTree.clear()
                values = None
                fitbool = None
                fitrange = None
                for an in range(nr_corrs(self)):
                    anSingle = file.get_analysis(an)
                    item = QTreeWidgetItem([anSingle.analysis_summary()])
                    analysisTree.addTopLevelItem(item)
                    fits = anSingle.fits
                    if an == file.active_analysis:
                        analysisTree.setCurrentItem(item)
                        if anSingle.corrs is None:
                            # no correlations have been calculated
                            fitrange = [1, 100]
                        else:
                            # correlation has been calculated, set fitrange to [1, max]
                            fitrange = [1, len(anSingle.get_corr())]
                    # add children
                    for f in range(len(fits)):
                        fit = fits[f]
                        # generate fit summary string
                        if fit.num_fitcurves > 0:
                            fitmodelname = fit.fit_all_curves[0].fitfunction_label
                            fitarray = fit.fit_all_curves[0].fitarray # always 12 values, e.g. [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
                            fitmodel = get_fit_model_from_name(fitmodelname)
                            if fitmodel is None:
                                outputString = 'Unknown fit model. ' + fitmodelname
                            else:
                                paramNames = fitmodel.param_names
                                paramInd = fitmodel.fitfunction_param_used
                                outputString = fitmodelname
                                j = 0
                                outputString += " (fit param: "
                                for ind in paramInd:
                                    if fitarray[ind]:
                                        outputString += paramNames[j][0:10] + ", "
                                    j += 1
                                outputString = outputString[:-2]
                                outputString += ")"
                            
                        itemc = QTreeWidgetItem([outputString])
                        item.addChild(itemc)
                        if f == anSingle.active_fit and an == file.active_analysis:
                            analysisTree.setCurrentItem(itemc)
                            # 11 values for the fit function edit fields
                            [startv, fitb, fitfunction] = fit.fitresults()
                            # set fitmodel dropdown to correct value
                            set_fit_modelbox(self, fitfunction)
                            # set fit values to fit results
                            if fitfunction == 'Maximum entropy method free diffusion':
                                Gsingle = anSingle.get_corr()
                                nparam = 6 if fitmodel.shortlabel == 'MEM free diff K' else 5
                                [fitresArrayDummy, tauD, startv, fitb] = fit.fitresults_mem(Gsingle[:,0], nparam)
                            if fitfunction in ['Model-free displacement analysis', 'Mean squared displacement']:
                                startv = fit.fit_all_curves[0].startvalues
                            values = startv
                            fitbool = fitb
                            fitrange = fit.fitrange()
                            
                # update fit box
                update_fit_model(self, values=values, fitbool=fitbool, fitrange=fitrange)
                
                # update diffusion law box
                update_difflaw_texts(self)
                
                # update time trace, finger print, correlation plot
                update_plots(self, [False, True, True, True], updatechunks=updatechunks)
            else:
                # uncolor this button
                fButtons[i].setStyleSheet("background-color:" + ap("fbcol"))
        else:
            if Newfilebutton == False:
                # add new button to select file
                fButtons[i].setText("New file")
                Newfilebutton = True
            else:
                fButtons[i].setText("File " + str(i+1))
            # uncolor this button
            fButtons[i].setStyleSheet("background-color:" + ap("fbcol"))
    if Nfiles == 0:
        analysisTree = self.ui.correlations_treeWidget
        analysisTree.clear()
        self.ui.FCSFolderName_label.setText(self.FCSFolderName)
        self.ui.FCSFolderName_label.setStyleSheet("color:" + ap("subtlecol"))
        self.ui.FCSFileName_label.setText(self.FCSFileName)
        update_fit_model(self)
        update_plots(self)


def update_active_button(self):
    im = self.ffslib.get_image()
    activeFile = im.active_ffs
    if activeFile is None:
        self.firstFile = 0
        self.activeFileButton = 0
    else:
        self.firstFile = np.max((activeFile - 4, 0))
        self.activeFileButton = activeFile - self.firstFile


class BrightEyesFFS(QMainWindow):
    
    def __init__(self, initialFile=None):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # QMainWindow.__init__(self)
        # if getattr(sys, 'frozen', False):
        #     RELATIVE_PATH = os.path.dirname(sys.executable)
        # else:
        #     RELATIVE_PATH = os.path.dirname('main.py')
        # self._ui_path = RELATIVE_PATH
        
        #loadUi(os.path.join(self._ui_path, 'brighteyes_ffs.ui'), self)
        
        self.setWindowTitle("BrightEyes-FFS")
        self.setWindowIcon(QIcon('files/ffs_icon.ico') )
        default_settings(self)
        update_buttons(self)
        
        ffsFilesButtons = file_buttons(self)
        for i in range(len(ffsFilesButtons)):
            button = ffsFilesButtons[i]
            button.clicked.connect(lambda state, x=i: self.activate_ffs_file(x))
        
        self.ui.prevImage_button.clicked.connect(self.previous_image)
        self.ui.nextImage_button.clicked.connect(self.next_image)
        self.ui.imageName_button.clicked.connect(self.change_image_file)
        
        self.ui.notes_edit.textChanged.connect(self.save_notes)
        self.ui.prevFCSfile_button.clicked.connect(self.previous_file)
        self.ui.nextFCSfile_button.clicked.connect(self.next_file)
        self.ui.saveLabel_button.clicked.connect(self.save_coords)
        self.ui.label_edit.textChanged.connect(self.save_label)
        
        
        self.ui.showElements_widget.clicked.connect(self.show_elements)
        self.ui.calcCorrCurrentFile_button.clicked.connect(self.add_corr_and_calc)
        self.ui.addToJoblist_button.clicked.connect(self.add_corr_to_joblist)
        self.ui.chunk_spinBox.valueChanged.connect(self.chunk_number_change)
        self.ui.chunkOn_checkBox.clicked.connect(self.chunk_checkbox_change)
        self.ui.showchunkscorr_dropdown.currentTextChanged.connect(self.show_chunks_corr_change)
        
        self.ui.actionOpen_image.triggered.connect(self.open_image_file)
        self.ui.actionChange_image.triggered.connect(self.change_image_file)
        self.ui.actionSave_session.triggered.connect(self.save_session)
        self.ui.actionSave_session_as.triggered.connect(self.save_session_as)
        self.ui.actionOpen_session.triggered.connect(self.open_session)
        self.ui.actionNew_session.triggered.connect(self.new_session)
        self.ui.actionExport_results_to_Excel.triggered.connect(self.export_session_xlsx)
        
        
        self.ui.actionCurrent_fit.triggered.connect(self.remove_current_fit)
        self.ui.actionCurrent_correlation.triggered.connect(self.remove_current_analysis)
        self.ui.actionCurrent_FFS_file.triggered.connect(self.remove_current_ffs_file)
        self.ui.actionCurrent_Image.triggered.connect(self.remove_current_image)
        
        self.ui.actionCopy_current_correlation.triggered.connect(self.copy_current_correlation)
        self.ui.actionCopy_current_fit_as_data.triggered.connect(self.use_current_fit_as_experimental_data)
        self.ui.actionPlot_in_Jupyter_Notebook.triggered.connect(self.plot_jupyter_notebook)
        self.ui.actionCreate_Jupyter_Notebook.triggered.connect(self.create_jupyter_notebook)
        self.ui.actionFilter_out_bad_chunks.triggered.connect(self.filter_out_bad_chunks)
        
        self.ui.correlations_treeWidget.clicked.connect(self.show_analysis)
        self.ui.actionCorrelation.triggered.connect(self.calc_corr_active_file)
        self.ui.actionAllCorrelationsCurrentFile.triggered.connect(self.calc_corr_all_files)
        self.ui.fitModel_dropdown.currentTextChanged.connect(self.show_fit_model)
        self.ui.newFit_button.clicked.connect(self.newfit)
        self.ui.overwriteFit_button.clicked.connect(self.overwritefit)
        self.ui.updateDiffLaw_button.clicked.connect(self.update_diffLaw)
        self.ui.updateDiffLawDiameter_button.clicked.connect(self.update_difflaw_diameter)
        self.ui.difflaw_dropdown.currentTextChanged.connect(self.difflaw_change)
        
       
        
        self.update_progress_bar(100)
        
        if initialFile != 'None' and initialFile is not None:
            
            open_ffs_file(self, 0, initialFile)
            
            self.activate_ffs_file(0)

    def activate_ffs_file(self, buttonNr):
        currentFileNr = self.firstFile + buttonNr
        Nfiles = nrfiles(self)
        if currentFileNr < Nfiles:
            # FFS file already loaded.
            self.activeFileButton = buttonNr
            im = self.ffslib.get_image()
            im.active_ffs = currentFileNr
        elif currentFileNr == Nfiles:
            # load new FFS file
            open_ffs_file(self, buttonNr)
        self.ui.chunk_spinBox.setValue(0)
        update_chunks(self)
        update_plots(self, updateAll = [True, False, False, False])
        update_buttons(self)
    
    def previous_image(self):
        try:
            imageNr = self.ffslib.active_image
            imageNr -= 1
            imageNr = np.max((imageNr, 0))
            self.ffslib.active_image = imageNr
            self.ui.chunk_spinBox.setValue(0)
            change_image(self)
        except:
            pass
    
    def next_image(self):
        try:
            imageNr = self.ffslib.active_image
            imageNr += 1
            imageNr = np.clip(imageNr, 0, self.ffslib.num_images-1)
            self.ffslib.active_image = imageNr
            self.ui.chunk_spinBox.setValue(0)
            change_image(self)
        except:
            pass
    
    def previous_file(self):
        buttonNr = self.activeFileButton - 1
        if buttonNr < 0:
            self.activeFileButton = 0
            self.firstFile = np.max((0, self.firstFile - 1))
        else:
            self.activeFileButton = buttonNr
        self.ui.chunk_spinBox.setValue(0)
        update_buttons(self)
    
    def next_file(self):
        buttonNr = self.activeFileButton
        newFileNr = self.firstFile + buttonNr + 1
        Nfiles = nrfiles(self)
        if Nfiles == 0:
            # no files yet
            self.firstFile = 0
            self.activeFileButton = 0
        elif newFileNr == Nfiles:
            # last file reached
            newFileNr -= 1
            self.firstFile = np.max((0, Nfiles - 4))
            self.activeFileButton = newFileNr - self.firstFile
        else:
            # go to next file
            if buttonNr < 4:
                self.activeFileButton += 1
            else:
                self.activeFileButton = 4
                self.firstFile += 1
        self.ui.chunk_spinBox.setValue(0)
        update_buttons(self)
    
    def save_label(self):
        # save label, (x,y) coordinates or both
        lb = self.ui.label_edit.text()
        currentFile = getfile(self)
        if currentFile is not None:
            currentFile.update(label=lb)
            update_buttons(self, updatechunks=False, updateOnlyButtonName=True)
        
    def save_coords(self):
        # save label, (x,y) coordinates or both
        lb = self.ui.label_edit.text()
        yc = self.ui.ycoord_edit.text()
        xc = self.ui.xcoord_edit.text()
        try:
            yc = int(yc)
        except:
            yc = 0 
        try:
            xc = int(xc)
        except:
            xc = 0 
        currentFile = getfile(self)
        if currentFile is not None:
            currentFile.update(label=lb, coords=[yc, xc])
            update_buttons(self)
            update_plots(self, updateAll=[True, False, False, False])
    
    def save_notes(self):
        notes = self.ui.notes_edit.toPlainText()
        self.ffslib.notes = notes
    
    def show_elements(self):
        file = getfile(self)
        update_timetrace(self, file)
    
    def chunk_number_change(self):
        print('chunk_number_change')
        # current chunk number was changed -> update checkbox
        QApplication.processEvents()
        update_chunks(self)
        QApplication.processEvents()
        # update time trace
        update_plots(self, updateAll=[False, True, False, False], updatechunks=False)
        # update corr plot if needed
        corrshow = str(self.ui.showchunkscorr_dropdown.currentText())
        if corrshow == "Show average all active chunks":
            pass
        else:
            update_plots(self, updateAll=[False, False, False, True], updatechunks=False)
    
    def show_chunks_corr_change(self):
        update_plots(self, updateAll=[False, False, False, True])
    
    def chunk_checkbox_change(self):
        # the "chunks-on" checkbox may change because the user clicked it
        # or because the user scrolled through the time trace
        # in the latter case, nothing needs to be done
        
        update_chunks_on(self)
    
    def show_analysis(self):
        # Check if top level item is selected or child selected
        item = self.ui.correlations_treeWidget.currentItem()
        if self.ui.correlations_treeWidget.indexOfTopLevelItem(item)==-1:
            anNr = self.ui.correlations_treeWidget.indexFromItem(item.parent()).row()
            fitNr = item.parent().indexOfChild(item)
        else:
            anNr = self.ui.correlations_treeWidget.currentIndex().row()
            fitNr = 'None'
        file = getfile(self)
        if file is not None:
            file.update(active_analysis=anNr)
            analysis = file.get_analysis()
            if analysis is not None:
                analysis.update(active_fit=fitNr)
            else:
                analysis.update(active_fit='None')
        else:
            file.update(active_analysis='None')
        update_buttons(self)
    
    def calc_corr_active_file(self):
        self.calc_correlations(calc='current')
    
    def add_corr_and_calc(self):
        addcorr_to_file(self)
        self.calc_corr_active_file()
    
    def calc_corr_all_files(self):
        self.calc_correlations(calc='all')
        
    def add_corr_to_joblist(self):
        addcorr_to_file(self)
    
    def open_image_file(self):
        self.update_progress_bar(0, message='Loading image...')
        fname = open_image_dialog()
        if fname is not None:
            self.finishedG = False
            self.update_progress_bar(50)
            th = threading.Thread(target=open_image_new_thread, args=(self, fname, 0))
            # Start the thread
            th.start()
            while not self.finishedG:
                QtTest.QTest.qWait(50)
                self.update_progress_bar(100 * self.progress, self.progressMessage)
            th.join()
            if self.imagetemp is not None:
                self.ffslib.add_image(self.imagetemp, self.fnametemp, load_ffs_metadata(fname))
                self.ffslib.active_image = self.ffslib.num_images - 1
                self.activeFileButton = 0
                plot_image(self)
                update_buttons(self)
        self.update_progress_bar(100, "Done.")
    
    def change_image_file(self):
        if self.ffslib.num_images > 0:
            imageNr = self.ffslib.active_image
            self.update_progress_bar(0, "Loading image...")
            fname = open_image_dialog()
            if fname is not None:
                self.finishedG = False
                self.update_progress_bar(50)
                th = threading.Thread(target=open_image_new_thread, args=(self, fname, 0))
                # Start the thread
                th.start()
                while not self.finishedG:
                    QtTest.QTest.qWait(50)
                    self.update_progress_bar(100 * self.progress)
                    self.ui.progressBar_label.setText(self.progressMessage)
                th.join()
                if self.imagetemp is not None:
                    imgObj = self.ffslib.get_image(imageNr)
                    imgObj.change_image(image=self.imagetemp, fname=self.fnametemp)
                    plot_image(self)
                    update_buttons(self)
            self.update_progress_bar(100, "Done.")
        else:
            self.open_image_file()
    
    def save_session(self, filepath=None):
        # use "filepath = ''" for saving to new file
        self.update_progress_bar(0)
        self.ui.progressBar_label.setText("")
        self.finishedG = False
        self.progress = 0
        if filepath is None or filepath is False:
            filepath = self.filePath
        th = threading.Thread(target=exportlib, args=(self, 0, filepath))
        # Start the thread
        th.start()
        while not self.finishedG:
            QtTest.QTest.qWait(50)
            self.update_progress_bar(100 * self.progress)
            self.ui.progressBar_label.setText(self.progressMessage)
        th.join()
        self.progressMessage = ""
        self.ui.progressBar_label.setText("Done.")
        self.update_progress_bar(100)
        self.setWindowTitle("BrightEyes-FFS - " + str(self.filePath))
    
    def save_session_as(self):
        print('save session as')
        self.save_session(filepath='')
    
    def open_session(self):
        self.update_progress_bar(0)
        self.ui.progressBar_label.setText("")
        self.finishedG = False
        self.progress = 0
        fname = open_ffslib()
        if fname is not None:
            self.filePath = fname
            th = threading.Thread(target=importlib, args=(self, fname, 0))
            # Start the thread
            th.start()
            while not self.finishedG:
                QtTest.QTest.qWait(50)
                self.update_progress_bar(100 * self.progress)
                self.ui.progressBar_label.setText(self.progressMessage)
            th.join()
            update_plots(self, updateAll=[True, False, False, False], updatechunks=True)
            update_active_button(self)
            update_buttons(self, updatechunks=False)
            self.setWindowTitle("BrightEyes-FFS - " + str(self.filePath))
            #update_plots(self)
        self.progressMessage = ""
        self.ui.progressBar_label.setText("Done.")
        self.update_progress_bar(100)
        
    
    def new_session(self):
        renew = showdialog('Create new session', 'Are you sure you want to create a new session? Unsaved changes will get lost.', '')
        if renew:
            clean_session(self)
    
    def export_session_xlsx(self):
        exportlib_xlsx(self)
    
    def calc_correlations(self, calc='all'):
        Nfiles = nrfiles(self)
        for i in range(Nfiles):
            # go through each file (of the current image) and calculate correlations
            file = getfile(self, i)
            # check if file still exists
            if calc=='all' or (i == self.firstFile+self.activeFileButton):
                [file.fname, fileFound, self.altFolderPath] = check_file_name(file.fname, self.altFolderPath)
            else:
                continue
            if fileFound:
                Nalayses = file.num_analyses
                for j in range(Nalayses):
                    self.update_progress_bar(0)
                    self.finishedG = False
                    self.progress = 0
                    analysis = file.get_analysis(j)
                    if analysis.calc_corr() and (calc=='all' or (i == self.firstFile+self.activeFileButton and j == file.active_analysis)):
                        # perform analysis
                        [anMode, anRes, anChsize] = analysis.corr_param()
                        anSettings = analysis.settings
                        startTime = time.time()
                        th = threading.Thread(target=calc_g_new_thread, args=(self, file, anSettings))
                        # Start the thread
                        th.start()
                        progressTxtStart = "Please wait, calculating correlations.\nThis may take a while."
                        self.ui.progressBar_label.setText(progressTxtStart)
                        percP = 0
                        eta = 1e5
                        while not self.finishedG:
                            QtTest.QTest.qWait(50)
                            percP = 100*self.progress # progress in %
                            self.update_progress_bar(percP)
                            timeLapsed = time.time() - startTime # s
                            neweta = (100 - percP) * timeLapsed / np.max((percP, 1e-5))
                            if neweta < 1:
                                eta = neweta
                            else:
                                eta = np.min((eta, neweta))
                            progressTxt = progressTxtStart
                            self.ui.progressBar_label.setText(progressTxt + "\nAbout " + time2string(eta) + " remaining.")
                        th.join()
                        if self.G is not None:
                            analysis.update(corrs = self.G)
                            if file.timetrace is None:
                                # first analysis of this file -> save airy and time trace
                                file.update(timetrace=self.data, airy=np.sum(self.data.astype(float), 0))
                update_buttons(self)
            self.ui.progressBar_label.setText("Done.")
            self.update_progress_bar(100)
    
    def use_current_fit_as_experimental_data(self):
        use_fit_as_data(self)
    
    def copy_current_correlation(self):
        copy_correlation(self)
        
    def plot_jupyter_notebook(self):
        fname = r'brighteyes_plot_saved_session_' + datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
        ffs_file = fname + '.ffs'
        self.save_session(filepath=ffs_file)
        notebook_file = fname + '.ipynb'
        plot_session_in_notebook(ffs_file, notebook_file)
        try:
            _ = subprocess.Popen(["jupyter", "notebook", notebook_file])
        except:
            showdialog('Error opening the notebook', 'The Jupyter Notebook has been created but could not be opened automatically. Please run Jupyter and open the notebook manually.', '')
    
    def create_jupyter_notebook(self):
        fname = r'brighteyes_saved_session_' + datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
        notebook_file = fname + '.ipynb'
        success = convert_session_to_notebook(self.ffslib, notebook_file)
        if success == 'success':
            try:
                _ = subprocess.Popen(["jupyter", "notebook", notebook_file])
            except:
                showdialog('Error opening the notebook', 'The Jupyter Notebook has been created but could not be opened automatically. Please run Jupyter and open the notebook manually.', '')
        else:
            showdialog('Error creating notebook', 'A Jupyter Notebook could not be created due to the following error: ' + success, '')
    
    def filter_out_bad_chunks(self):
        update_chunks_on_filter(self)
    
    def remove_current_fit(self):
        remove_fit(self)
    
    def remove_current_analysis(self):
        remove_analysis(self)
    
    def remove_current_ffs_file(self):
        remove_ffs_file(self)
    
    def remove_current_image(self):
        remove_image(self)
    
    def show_fit_model(self):
        update_fit_model(self)
    
    def newfit(self):
        self.update_progress_bar(0, 'Updating fit analysis...')
        update_fit_analysis(self, ftype='new')
        self.update_progress_bar(100, '')
    
    def overwritefit(self):
        self.update_progress_bar(0, 'Updating fit analysis...')
        update_fit_analysis(self, ftype='update')
        self.update_progress_bar(100, '')
    
    def update_diffLaw(self):
        update_w0_diff(self)
    
    def update_difflaw_diameter(self):
        update_diameter(self)
    
    def difflaw_change(self):
        plot_difflaw(self)
    
    def update_progress_bar(self, value, message=None):
        self.ui.progressBar.setValue(int(value))
        if message is not None:
            self.ui.progressBar_label.setText(message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='None', help='Absolute file path to .h5 file')
    args = parser.parse_args()

    app = QApplication([])
    app.setWindowIcon(QIcon("files/ffs_icon.ico"))  # Set the application icon here
        
    pixmap = QPixmap("files/brighteyes_ffs_startup_splash.png")
    splash = QSplashScreen(pixmap)
    # LICENSE = (
    #         """BrightEyes-FFS (Version: %s)                  
    #     Author: Eli Slenders | MMS 
    #     License: General Public License version 3 (GPL v3)        
    #     Copyright © 2024 Istituto Italiano di Tecnologia
    #     """)

    # splash.showMessage(
    #             LICENSE,
    #             QtCore.Qt.AlignTop | QtCore.Qt.AlignRight
    #         )

    splash.show()
    app.processEvents()
    
    app.setStyleSheet(qdarkstyle.load_stylesheet(palette=qdarkstyle.light.palette.LightPalette))
    window = BrightEyesFFS(args.file)
    window.setWindowIcon(QIcon("files/ffs_icon.ico"))  # Set the window icon after creating the main window
    
    window.show()
    splash.finish(window)
    sys.exit(app.exec_())
    
    