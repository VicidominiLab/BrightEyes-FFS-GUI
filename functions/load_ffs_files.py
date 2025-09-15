from PyQt5.QtWidgets import QFileDialog, QDialog
import numpy as np
import h5py
import os.path as ospath
import tifffile

from brighteyes_ffs.fcs.meas_to_count import file_to_count, reshape_to_5d
from brighteyes_ffs.tools.path2fname import path2fname

def open_image_dialog():
    """
    Open image .bin or .h5 file
    dialog window is opened to select file
    file name is returned
    """
    
    ftype = "Images (*.tiff *.tif *.bin *.h5)"
    fname = open_dialog('Open image file', ftype)
    
    return fname

def open_image(fname, root=0):
    """
    Open image .bin, .tiff or .h5 file
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    fname       String with path to .bin or .h5 image file
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    image       2D image with sum5x5, summed over all time bins. In case of a
                z stack, the central image is taken. If nothing is selected,
                None is returned
    fname       input file name
    ===========================================================================
    """
    if fname is None:
        return None, None
    
    if fname[-4:] == ".bin":
        if root != 0:
            root.progress = 0
            root.progressMessage = 'Opening bin file...'
        [out, frames, y, x, time_per_pixel] = file_to_count(fname, print_info=False)
        
        if root != 0:
            root.progress = 0.50
            root.progressMessage = 'Building image...'
        image = reshape_to_5d(out, frames, y, x, time_per_pixel)
        Nframes = np.shape(image)[0]
        plotframe = int(np.floor(Nframes/2))
        image = np.sum(np.sum(image[plotframe, :,:,:,:], 3), 2) # sum over all sensors and times
        image = np.squeeze(image)
        if root != 0:
            root.progress = 0.90
            root.progressMessage = 'Almost there...'
            
    elif fname[-3:] == ".h5":
        if root != 0:
            root.progress = 0
            root.progressMessage = 'Opening h5 file...'
        
        with h5py.File(fname, "r") as f:
            key = None
            keys = list(f.keys())
            Nkeys = len(keys)
            if Nkeys == 1:
                key = keys[0]
            else:
                for i in range(Nkeys):
                    if keys[i].lower() == 'data':
                        key = keys[i]
                        break
                    if 'data' in keys[i].lower() and 'meta' not in keys[i].lower() and 'analog' not in keys[i].lower():
                        key = keys[i]
            if key is None:
                if root != 0:
                    root.progress = 0.90
                    root.progressMessage = 'No data set found.'
                return None, None
            
            # Data set found. Get the data Tzyxtc
            if root != 0:
                root.progress = 0.50
                root.progressMessage = 'Data set with key ' + key + ' found. Loading data...'
            data = f[key]
            image = np.squeeze(data)
            Ndim = len(np.shape(image))
            while Ndim > 2:
                image = np.sum(image, np.argmin(np.shape(image)))
                Ndim = len(np.shape(image))
            if root != 0:
                root.progress = 0.90
                root.progressMessage = 'Almost there...'
                
    elif fname[-4:] == ".tif" or fname[-5:] == ".tiff":
        im = tifffile.imread(fname)
        sorted_dims = sorted(enumerate(im.shape), key=lambda x: x[1], reverse=True)
        largest_dims = [sorted_dims[0][0], sorted_dims[1][0]]
        axes_to_sum = tuple(i for i in range(im.ndim) if i not in largest_dims)
        summed_image = np.sum(im, axis=axes_to_sum)
        image = summed_image
    
    else:
        image = None
        fname = None
                    
    return image, fname

def open_ffs(fname=''):
    """
    Select FFS .h5 file (or tiff, czi, bin)
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    dialog window is opened to select file
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    the file name is returned (without loading the actual file)
    ===========================================================================
    """
    
    ftype = "FFS file (*.bin *.h5 *.tiff *.tif *.czi *.csv)"
    fname = open_dialog('Select FFS file ' + fname, ftype, '/')
    
    if fname != "":
        return fname
    else:
        return None

def open_ffslib():
    """
    Select FFS .lib project file
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    dialog window is opened to select file
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    the file name is returned (without loading the actual file)
    ===========================================================================
    """
    
    ftype = "FFS session (*.ffs *.ffz)"
    fname = open_dialog('Select FFS lib file', ftype, '/')
    
    if fname != "":
        return fname
    else:
        return None

def save_ffs(windowTitle='Save project as', ftype='FFS Files (*.ffs)', directory=''):
    """
    Select name to save ffs lib
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    dialog window is opened to choose file name
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    the file name is returned
    ===========================================================================
    """
    
    fname, _ = QFileDialog.getSaveFileName(None, windowTitle, directory, ftype)

    return fname if fname else None

def check_file_exists(file, defaultFolder=""):
    """
    Check if file exists. If not, check in defaultFolder.
    If not, open dialog to select new file
    ===========================================================================
    Input           Meaning
    ---------------------------------------------------------------------------
    file            Path to .ffs file (full path)
    defaultFolder   Path to alternative folder
    ===========================================================================
    Output          Meaning
    ---------------------------------------------------------------------------
    file            Path to file that exists (full path)
    newdefaultFolder New folder where the file was found
    ===========================================================================
    """
    [fname, newdefaultFolder] = path2fname(file, properWay=True)
    
    if ospath.exists(file):
        return file, newdefaultFolder
    
    file = ospath.join(defaultFolder, fname)
    if ospath.exists(file):
        return file, defaultFolder
    
    file = open_ffs(fname)
    if file is not None:
        [file, newdefaultFolder] = path2fname(file, properWay=True)
        return ospath.join(newdefaultFolder, file), newdefaultFolder
    
    return None, defaultFolder


def check_file_name(fname, defFolder):
    fileFound = True
    [file, deffolder] = check_file_exists(fname, defFolder)
    if file is None:
        file = fname
        fileFound = False
    return file, fileFound, deffolder


def open_dialog(windowTitle='Open file', ftype='*.bin', directory=''):
    dialog = QFileDialog()
    dialog.setWindowTitle(windowTitle)
    dialog.setNameFilter(ftype)
    dialog.setDirectory(directory)
    dialog.setFileMode(QFileDialog.ExistingFile)
    
    filename = None
    fname = None
    if dialog.exec_() == QDialog.Accepted:
        filename = dialog.selectedFiles()
    if filename:
        fname = str(filename[0])
    
    return fname