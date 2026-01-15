import os
from brighteyes_ffs.tools import list_files
from brighteyes_ffs.fcs_gui.correlation_functions_class import CorrelationFunction
import sys

IS_FROZEN = getattr(sys, 'frozen', False)

def list_of_correlation_functions():
    listOfCorrelations = []
    
    if IS_FROZEN:
        # manual import
        
        from functions.correlations.all_autocorrelations import get_params
        params = get_params()
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        listOfCorrelations.append(modename)
        
        from functions.correlations.all_crosscorrelations import get_params
        params = get_params()
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        listOfCorrelations.append(modename)
        
        from functions.correlations.asymmetric_diffusion import get_params
        params = get_params()
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        listOfCorrelations.append(modename)
        
        from functions.correlations.cross_center import get_params
        params = get_params()
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        listOfCorrelations.append(modename)
        
        from functions.correlations.cross_correlation_spectroscopy import get_params
        params = get_params()
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        listOfCorrelations.append(modename)
        
        from functions.correlations.custom import get_params
        params = get_params()
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        listOfCorrelations.append(modename)
        
        from functions.correlations.pair_correlation_fcs import get_params
        params = get_params()
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        listOfCorrelations.append(modename)
        
        from functions.correlations.single_element_fcs import get_params
        params = get_params()
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        listOfCorrelations.append(modename)
        
        from functions.correlations.spot_variation_fcs import get_params
        params = get_params()
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        listOfCorrelations.append(modename)
        
        from functions.correlations.spot_variation_fcs_extended import get_params
        params = get_params()
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        listOfCorrelations.append(modename)
        
    else:
    
        files = list_files('functions/correlations/', 'py')
        for f in files:
            if '__' not in f:
                fctn = os.path.basename(f)
                fctn = fctn[:-3]
                
                crltn = __import__(f'functions.correlations.{fctn}', fromlist=[fctn])
                params = crltn.get_params()
                
                # Instantiate the class
                correlation = CorrelationFunction()
                correlation.set_params(params)
                
                modename = correlation.mode
                if modename != 'Curve from file':
                    listOfCorrelations.append(modename)
                
    correlationsSpotVar = [i for i in listOfCorrelations if i.startswith("Spot-variation")]
    correlationsOther = [i for i in listOfCorrelations if i not in correlationsSpotVar]
    listOfCorrelations = correlationsSpotVar + correlationsOther
    return listOfCorrelations


def get_correlation_object_from_name(name, det_type='Square 5x5'):
    
    if IS_FROZEN:
        # manual import
        
        from functions.correlations.all_autocorrelations import get_params
        params = get_params(det_type)
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        if name == modename:
            return correlation
        
        from functions.correlations.all_crosscorrelations import get_params
        params = get_params(det_type)
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        if name == modename:
            return correlation
        
        from functions.correlations.asymmetric_diffusion import get_params
        params = get_params(det_type)
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        if name == modename:
            return correlation
        
        from functions.correlations.cross_center import get_params
        params = get_params(det_type)
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        if name == modename:
            return correlation
        
        from functions.correlations.cross_correlation_spectroscopy import get_params
        params = get_params(det_type)
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        if name == modename:
            return correlation
        
        from functions.correlations.custom import get_params
        params = get_params(det_type)
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        if name == modename:
            return correlation
        
        from functions.correlations.pair_correlation_fcs import get_params
        params = get_params(det_type)
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        if name == modename:
            return correlation
        
        from functions.correlations.single_element_fcs import get_params
        params = get_params(det_type)
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        if name == modename:
            return correlation
        
        from functions.correlations.spot_variation_fcs import get_params
        params = get_params(det_type)
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        if name == modename:
            return correlation
        
        from functions.correlations.spot_variation_fcs_extended import get_params
        params = get_params(det_type)
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        if name == modename:
            return correlation
        
        from functions.correlations.corr_from_csv import get_params
        params = get_params(det_type)
        correlation = CorrelationFunction()
        correlation.set_params(params)
        modename = correlation.mode
        if name == modename:
            return correlation
        
    else:
    
        files = list_files('functions/correlations/', 'py')
        for f in files:
            if '__' not in f:
                fctn = os.path.basename(f)
                fctn = fctn[:-3]
                crltn = __import__(f'functions.correlations.{fctn}', fromlist=[fctn])
                params = crltn.get_params(det_type)
                
                # Instantiate the class
                correlation = CorrelationFunction()
                correlation.set_params(params)
                
                modename = correlation.mode
                if name == modename:
                    return correlation
        return None