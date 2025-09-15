import nbformat as nbf
import numpy as np
from .fitmodels import get_fit_model_from_name

def plot_session_in_notebook(ffs_path=r'D:\\brighteyes_saved_session.ffs', notebook_path="D:\\generated_notebook.ipynb"):
    # Create a new notebook object
    nb = nbf.v4.new_notebook()

    # Create markdown and code cells
    cells = [
        nbf.v4.new_markdown_cell("# Generated Jupyter Notebook"),
        nbf.v4.new_markdown_cell("Import packages"),
        nbf.v4.new_code_cell("""from brighteyes_ffs.fcs_gui.read_ffs import read_g_from_ffs, read_ffs_file, restorelib, read_fitresults_from_ffs
from brighteyes_ffs.tools.color_from_map import color_from_map
import matplotlib.pyplot as plt
import numpy as np"""),
        nbf.v4.new_markdown_cell("Load session from GUI"),
        nbf.v4.new_code_cell("""ffs_file = \"""" + ffs_path + """\""""),
        nbf.v4.new_markdown_cell("Plot image"),
        nbf.v4.new_code_cell("""ffslib = restorelib(ffs_file) # read .ffs file
im_nr = ffslib.active_image    # get image number of open image
im = ffslib.get_image(im_nr)   # get image from object"""),
        nbf.v4.new_code_cell("""plt.figure()
plt.imshow(im.image, cmap='hot')
k = plt.xticks([])
k = plt.yticks([])

for i in range(im.num_files):
    ffs = im.ffs_list[i]
    c = ffs.coords
    plt.scatter(c[1], c[0], color='white')"""),
        nbf.v4.new_markdown_cell("Plot time trace"),
        nbf.v4.new_code_cell("""ffs = im.ffs_list[im.active_ffs]
md = ffs.metadata
duration = md.duration # s"""),
        nbf.v4.new_code_cell("""plt.figure()
for i in range(25):
    y = ffs.timetrace[:,i]
    bin_time = duration / len(y)
    t = np.arange(0, duration, bin_time)    
    plt.plot(t, y/bin_time/1000)
plt.xlabel('Time (s)')
l = plt.ylabel('Photon count rate (kHz)')"""),
        nbf.v4.new_markdown_cell("""Extract the correlations and fits that are currently open in the GUI  
Use read=[a, b, c, d] to read a different correlation and fit. Here,  
a = an integer indicating which image number, usually 0  
b = an integer indicating which file to read  
c = an integer indicating which correlation to read from that file  
d = an integer indicating which fit to read from that correlation"""),
        nbf.v4.new_code_cell("""analysis = read_ffs_file(ffs_file, read='active', returnObj='analysis')
algorithm = analysis.settings.algorithm
G, tau, Gfit, taufit = read_g_from_ffs(ffs_file, read='active')"""),
        nbf.v4.new_code_cell("""plt.figure()
for i in range(np.shape(G)[1]):
    if algorithm == 'pch':
        plt.bar(tau, G[:,i], alpha=0.5, color=color_from_map(np.mod(i, 8), startv=0, stopv=8, cmap='Set2'))
    else:
        plt.scatter(tau, G[:,i], s=20, alpha=0.5, color=color_from_map(np.mod(i, 8), startv=0, stopv=8, cmap='Set2'))
    plt.plot(taufit, Gfit[:,i], color=color_from_map(np.mod(i, 8), startv=0, stopv=8, cmap='Set2'))

if algorithm == 'pch':
    plt.ylim([0,1.05*np.max(G[0:,:])])
    plt.xlabel('Counts')
    plt.ylabel('Relative frequency')
    plt.xscale('linear')
else:
    plt.ylim([0,1.05*np.max(G[1:,:])])
    plt.xlabel('Lag time (s)')
    plt.ylabel('G')
    plt.xscale('log')"""),
        nbf.v4.new_markdown_cell("Fit results"),
        nbf.v4.new_code_cell("""print(read_fitresults_from_ffs(ffs_file))"""),
        nbf.v4.new_code_cell("""# For MEM analysis, use
fitmem, tauD = read_fitresults_from_ffs(ffs_file)
plt.figure()
for i in range(len(fitmem)):
    plt.scatter(tauD, fitmem[i], s=20, alpha=0.5, color=color_from_map(np.mod(i, 8), startv=0, stopv=8, cmap='Set2'))
    plt.plot(tauD, fitmem[i], color=color_from_map(np.mod(i, 8), startv=0, stopv=8, cmap='Set2'))
plt.xscale('log')
plt.xlabel('Diffusion time (s)')
plt.ylabel('Relative concentration')"""),
        
    ]

    # Add the cells to the notebook
    nb['cells'] = cells

    # Write the notebook to a file
    with open(notebook_path, "w") as f:
        nbf.write(nb, f)


def convert_session_to_notebook(lib, notebook_path):
    # check image
    if lib.active_image is None:
        return "No active image found"
    image_object = lib.lib[lib.active_image]
    image_path = image_object.image_name
    
    # check correlation file
    if image_object.active_ffs is None or image_object.get_ffs_file() is None:
        return "No active FFS file found"
    ffs_file_object = image_object.get_ffs_file()
    ffs_file_path = ffs_file_object.fname
    ffs_file_label = ffs_file_object.label
    ffs_file_object.analysis_list
    
    # check analysis
    analysis_object = ffs_file_object.get_analysis()
    list_of_g = analysis_object.settings.list_of_g
    list_of_g_str = list_to_string(list_of_g)
    list_of_g_out_str = list_to_string(analysis_object.settings.elements)
    resolution = str(int(analysis_object.settings.resolution))
    algorithm = str(analysis_object.settings.algorithm)
    chunksize = str(analysis_object.settings.chunksize)
    chunks_off = analysis_object.settings.chunks_off
    chunks_idx = list(np.nonzero(chunks_off)[0])
    chunks_off_str = list_to_string(chunks_idx, element_type="")
    
    # Create a new notebook object
    nb = nbf.v4.new_notebook()

    # Create markdown and code cells
    cells = [
        nbf.v4.new_markdown_cell("# Generated Jupyter Notebook"),
        nbf.v4.new_markdown_cell("Import packages"),
        nbf.v4.new_code_cell("""from brighteyes_ffs.fcs.fcs2corr import fcs_load_and_corr_split as correlate
from brighteyes_ffs.fcs.fcs2corr import fcs_av_chunks
from brighteyes_ffs.fcs.fcs_fit import fcs_fit
from brighteyes_ffs.fcs.get_fcs_info import get_metafile_from_file, get_file_info
from brighteyes_ffs.fcs_gui.read_ffs import read_g_from_ffs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np"""),
        nbf.v4.new_markdown_cell("File name and correlation settings"),
        nbf.v4.new_code_cell("""file = \"""" + ffs_file_path + """"
list_of_g = """ + list_of_g_str +  """
list_of_g_out = """ + list_of_g_out_str +  """
algorithm = '""" + algorithm + """'
resolution = """ + resolution + """
chunksize = """ + chunksize),
        nbf.v4.new_code_cell("""mdata = get_file_info(get_metafile_from_file(file))"""),
        nbf.v4.new_markdown_cell("Calculate correlations"),
        nbf.v4.new_code_cell("""G, time_trace = correlate(file, list_of_g=list_of_g, accuracy=resolution, split=chunksize, time_trace=True, list_of_g_out=list_of_g_out, algorithm=algorithm)"""),
        nbf.v4.new_markdown_cell("Plot time trace"),
        nbf.v4.new_code_cell("""num_chunks = int(np.floor(mdata.duration / chunksize))
splits = np.arange(0, (num_chunks+1)*chunksize, chunksize)
good_chunks = """ + chunks_off_str ),
        nbf.v4.new_code_cell("""fig, ax = plt.subplots()
for i in range(25):
    plt.plot(time_trace[:,i])
ymin = np.min(time_trace)
ymax = np.max(time_trace)
if len(splits) <= 101:
    # color chunks red if not used for calculating average correlation
    if good_chunks is not None:
        for i in range(len(splits) - 1):
            if i not in good_chunks:
                rect = patches.Rectangle((10*splits[i], ymin), 10*(splits[i+1]-splits[i]), ymax-ymin, fc='r')
                ax.add_patch(rect)
plt.xlabel('Time (arb. units)')
plt.ylabel('Photon counts per bin')"""),
        nbf.v4.new_markdown_cell("""Plot correlations  
First, calculate the average correlation of the good chunks. Then, plot the result."""),
        nbf.v4.new_code_cell("""G = fcs_av_chunks(G, good_chunks)"""),
        
        nbf.v4.new_code_cell("""plt.figure()
for corr in list_of_g_out:
    Gsingle = getattr(G, corr + '_averageX')
    plt.scatter(Gsingle[1:,0], Gsingle[1:,1], s=4, label=corr)
plt.legend()
plt.xlabel('Lag time (s)')
plt.ylabel('G')
plt.xscale('log')"""),
]
    if analysis_object.active_fit is not None:
        # check fit
        fit_object = analysis_object.return_fit_obj()
        fit_obj_single = fit_object.fit_all_curves[0]
        minbound = list_to_string(fit_obj_single.minbound)
        maxbound = list_to_string(fit_obj_single.maxbound)
        paramFactors10 = list_to_string(fit_obj_single.param_factors10)
        fitfunction = fit_obj_single.fitfunction
        
        if fitfunction is not None:
        
            fitfunction_label = fit_obj_single.fitfunction_label # more readable name of the fit function
            fitrange = fit_obj_single.fitrange
            fitstart = str(int(fitrange[0]))
            fitstop = str(int(fitrange[1]))
            fit_info = list_to_string(fit_obj_single.fitarray[:-1], element_type="")
            fit_startv = list_to_string(fit_obj_single.startvalues, element_type="")
            fitmodel = get_fit_model_from_name(fit_obj_single.fitfunction_label)
            fitf = str(fitmodel.fitfunction_name)
            
            cells += [
            nbf.v4.new_markdown_cell("Fit correlations"),
            nbf.v4.new_code_cell("""from """ + str(fitfunction.__module__) + """ import """ + fitfunction.__name__ + """ as my_fit_fun # you chose """ + fitfunction_label + """ as the fit function""" ),
            nbf.v4.new_code_cell("""fitresults = []
for corr in list_of_g_out:
    Gsingle = getattr(G, corr + '_averageX')
    Gexp = Gsingle[""" + fitstart + """:""" + fitstop + """,1] # your chosen range for fitting
    tau = Gsingle[""" + fitstart + """:""" + fitstop + """,0] # corresponding tau values
    fit_info = np.asarray(""" + fit_info + """) # use 1 for parameters that have to be fitted, 0 otherwise
    param = np.asarray(""" + fit_startv + """)
    lBounds = np.asarray(""" + minbound + """) # the lower bounds for the fit parameters
    uBounds = np.asarray(""" + maxbound + """) # the upper bounds for the fit parameters
    fitresult = fcs_fit(Gexp, tau, my_fit_fun, fit_info, param, lBounds, uBounds, plotInfo=-1)
    fitresults.append(fitresult)"""),
    nbf.v4.new_code_cell("""plt.figure()
for i, corr in enumerate(list_of_g_out):
    # plot correlation
    Gsingle = getattr(G, corr + '_averageX')
    plt.scatter(Gsingle[""" + fitstart + """:""" + fitstop + """,0], Gsingle[""" + fitstart + """:""" + fitstop + """,1], s=4, label=corr)
    # plot fit
    fitresult = fitresults[i]
    plt.plot(Gsingle[""" + fitstart + """:""" + fitstop + """,0], Gsingle[""" + fitstart + """:""" + fitstop + """,1]-fitresult.fun)
plt.legend()
plt.xlabel('Lag time (s)')
plt.ylabel('G')
plt.xscale('log')"""),
            ]

        if fit_obj_single.fitfunction_label == 'Asymmetry heat map':
            cells += [
            nbf.v4.new_markdown_cell("Asymmetry heat map"),
            nbf.v4.new_code_cell("""from brighteyes_ffs.fcs.fcs_polar import g2polar""" ),
            nbf.v4.new_code_cell("""num_curves = len(list_of_g_out)
if num_curves == 4:
    columnorder = ['Right', 'Up', 'Left', 'Down'] # square array detector
elif num_curves == 6:
    columnorder = ['Right', 'UpRight', 'UpLeft', 'Left', 'DownLeft', 'DownRight'] # airy detector
else:
    print("something went wrong")

N = len(getattr(G, columnorder[0] + '_averageX')[:,1])
allfits = np.zeros((N, len(columnorder)))
for i in range(num_curves):
    allfits[:,i] = getattr(G, columnorder[i] + '_averageX')[:,1]

z = g2polar(allfits[1:,:])
R = len(z) / 2
phi = np.linspace(0, 2*np.pi, 360)
plt.figure()
plt.imshow(np.flipud(z), cmap='jet')
plt.plot(R*np.cos(phi) + R, R*np.sin(phi)+R, '-', color='k', linewidth=5)
plt.xlim([-0.1*R, 2.1*R])
plt.ylim([-0.1*R, 2.1*R])"""),
        
        ]
        

    # Add the cells to the notebook
    nb['cells'] = cells

    # Write the notebook to a file
    with open(notebook_path, "w") as f:
        nbf.write(nb, f)
    
    return "success"


def list_to_string(my_list, element_type="'"):
    list_str = "["
    for i in my_list:
        list_str += element_type + str(i) + element_type + ", "
    list_str = list_str[:-2] + "]"
    return list_str
    