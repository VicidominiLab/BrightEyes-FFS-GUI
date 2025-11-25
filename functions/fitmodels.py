import numpy as np
import re
import os
import sys
from brighteyes_ffs.tools import list_files
from brighteyes_ffs.fcs_gui.fitmodel_class import FitModel
import importlib

IS_FROZEN = getattr(sys, 'frozen', False)

#from .fitfunctions import *

# from .fitfunctions.free_diffusion_1_component import free_diffusion_1_component
# from .fitfunctions.free_diffusion_1_component_finite_length import free_diffusion_1_component_finite_length
# from .fitfunctions.free_diffusion_1_component_flow import free_diffusion_1_component_flow
# from .fitfunctions.free_diffusion_2_components import free_diffusion_2_components
# from .fitfunctions.free_diffusion_circular_scanning import free_diffusion_circular_scanning
# from .fitfunctions.free_diffusion_pair_correlation import free_diffusion_pair_correlation
# from .fitfunctions.mem_free_diffusion import mem_free_diffusion
# from .fitfunctions.model_free_displacement_analysis import model_free_displacement_analysis
# from .fitfunctions.two_components_with_afterpulsing import two_components_with_afterpulsing
# from .fitfunctions.anomalous_diffusion_1_component import anomalous_diffusion_1_component
# from .fitfunctions.flow_heat_map import flow_heat_map

#from . import fitfunctions as ff


def convert_string_2_start_values(startvaluesList, mode):
    Nparam = len(startvaluesList)
    if type(mode) == int:
        startArray = np.zeros((Nparam, mode)) + 1
        Ncurves = mode
    elif mode == "Spot-variation fcs":
        startArray = np.zeros((Nparam, 3)) + 1 # central, sum3, sum5
        Ncurves = 3
    elif mode == "Cross-correlation spectroscopy":
        startArray = np.zeros((Nparam, 4)) + 1 # cross12, cross21, auto1, auto2
        Ncurves = 4
    elif mode == "All autocorrelations":
        startArray = np.zeros((Nparam, 25)) + 1 # 25 autocorrelations: det0x0, det1x1 etc.
        Ncurves = 25
    elif mode == "Custom":
        startArray = np.zeros((Nparam, 1)) + 1 # 25 autocorrelations: det0x0, det1x1 etc.
        Ncurves = 25
    else:
        return None
    for i in range(Nparam):
        v = string_2_float_list(startvaluesList[i]) # e.g. "1.0, 2.1, 3.3" -> [1.0, 2.1, 3.3]
        if len(v) == 1:
            # only one value given for all curves
            startArray[i, :] = v[0]
        else:
            for j in range(np.min((len(v), Ncurves))):
                startArray[i, j] = v[j]
    return startArray


def string_2_float_list(string):
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    floatList = rx.findall(string)
    floatList = [float(i) for i in floatList]
    return floatList

def list_of_fit_models():
    fitmodels = []
    if IS_FROZEN:
        # manual import
        
        from functions.fitfunctions.anomalous_diffusion_1_component import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.anomalous_diffusion_2_components import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.asymmetry_heat_map import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.flow_heat_map import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.free_diffusion_1_component import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.free_diffusion_1_component_finite_length import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.free_diffusion_1_component_flow import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.free_diffusion_2_components import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.free_diffusion_2_components_2d import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.free_diffusion_circular_scanning import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.free_diffusion_flow_global_fit import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.free_diffusion_pair_correlation import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.mean_squared_displacement import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.mem_free_diffusion import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.model_free_displacement_analysis import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.nanosecond_fcs import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.two_components_with_afterpulsing import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.uncoupled_reaction_diffusion import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.pch_1_component import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.pch_2_components import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
        from functions.fitfunctions.pch_2_components_global import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        fitmodels.append(modelname)
        
    else:
        files = list_files('functions/fitfunctions/', 'py')
        for f in files:
            if '__' not in f:
                # dynamic import
                fctn = os.path.basename(f)
                fctn = fctn[:-3]
                fitf = importlib.import_module(f'functions.fitfunctions.{fctn}')
                params = fitf.get_params()
                fitmodel = FitModel()
                fitmodel.set_params(params)
                modelname = fitmodel.model
                fitmodels.append(modelname)
                
    fitmodelsFree = [i for i in fitmodels if i.startswith("Free diffusion") and '2D' not in i]
    fitmodels2D = [i for i in fitmodels if '2D' in i and i not in fitmodelsFree]
    fitmodelsOther = [i for i in fitmodels if i not in fitmodelsFree and i not in fitmodels2D]
    fitmodels = fitmodelsFree + fitmodels2D + fitmodelsOther
    return fitmodels

def get_fit_model_from_name(name):
    if IS_FROZEN:
        # manual import
        
        from functions.fitfunctions.anomalous_diffusion_1_component import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.anomalous_diffusion_2_components import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.asymmetry_heat_map import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.flow_heat_map import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
    
        from functions.fitfunctions.free_diffusion_1_component import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.free_diffusion_1_component_2d import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.free_diffusion_1_component_finite_length import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.free_diffusion_1_component_flow import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.free_diffusion_2_components import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.free_diffusion_2_components_2d import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.free_diffusion_circular_scanning import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.free_diffusion_flow_global_fit import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.free_diffusion_pair_correlation import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.mean_squared_displacement import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.mem_free_diffusion import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.model_free_displacement_analysis import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.nanosecond_fcs import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.two_components_with_afterpulsing import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.uncoupled_reaction_diffusion import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.pch_1_component import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.pch_2_components import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
        from functions.fitfunctions.pch_2_components_global import get_params
        params = get_params()
        fitmodel = FitModel()
        fitmodel.set_params(params)
        modelname = fitmodel.model
        if name == modelname:
            return fitmodel
        
    else:
        files = list_files('functions/fitfunctions/', 'py')
        for f in files:
            if '__' not in f:
                # dynamic import
                fctn = os.path.basename(f)
                fctn = fctn[:-3]
                fitf = importlib.import_module(f'functions.fitfunctions.{fctn}')
                params = fitf.get_params()
                fitmodel = FitModel()
                fitmodel.set_params(params)
                modelname = fitmodel.model
                if name == modelname:
                    return fitmodel
    return None