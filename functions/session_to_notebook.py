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
from brighteyes_ffs.fcs_gui.timetrace_end import timetrace_end
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
for i in range(len(ffs.timetrace[0,:])):
    last_idx = timetrace_end(ffs.timetrace)
    y = ffs.timetrace[0:last_idx,i]
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
        nbf.v4.new_code_cell("""# For flow heat map, use
fit = read_ffs_file(ffs_file, read='active', returnObj='fit')
heatmap, arrow, columnNotFound = fit.fitresults_flowmap()
plt.figure()
plt.imshow(np.flipud(heatmap), cmap='PiYG')
phi = np.linspace(0, 2*np.pi, 360)
R = len(heatmap) / 2
plt.plot(R*np.cos(phi) + R, R*np.sin(phi)+R, '-', color='k', linewidth=5)
r = arrow[0]
u = arrow[1]
plt.arrow(90-(r/2), 90-(u/2), r, u, width=1, head_width=4, color='k', length_includes_head=True)
plt.xlim([-0.1*R, 2.1*R])
plt.ylim([-0.1*R, 2.1*R])
plt.axis('off')"""),
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
    ffs_file_path = ffs_file_object.fname.replace('\\', '/')
    ffs_file_label = ffs_file_object.label
    ffs_file_object.analysis_list
    
    # check analysis
    analysis_object = ffs_file_object.get_analysis()
    list_of_g = analysis_object.settings.list_of_g
    list_of_g_str = list_to_string(list_of_g)
    list_of_g_out_str = list_to_string(analysis_object.settings.elements)
    resolution = str(int(analysis_object.settings.resolution))
    averaging = analysis_object.settings.average
    averaging_str = 'None'
    if averaging is not None:
        averaging_str = list_to_string(averaging)
    algorithm = str(analysis_object.settings.algorithm)
    if algorithm == 'pch':
        import_corr = 'from brighteyes_ffs.fcs.fcs2corr import fcs_load_and_corr_split as correlate'
        algorithm_settings_str = 'accuracy=30, binsize=resolution'
        algorithm_str = ', algorithm=algorithm'
        xlabel = 'Counts per bin'
        ylabel = 'Relative frequency'
        xscale = 'linear'
        yscale = 'log'
        fit_str0 = 'nparam = len(param) - 3'
        fit_str1 = 'fitresult = fit_pch(Gexp, fit_info[0:nparam], param[0:nparam], psf=list(param[nparam:nparam+2]), fitfun=my_fit_fun, lBounds=lBounds[0:nparam], uBounds=uBounds[0:nparam], n_bins=param[nparam+2], minimization="absolute")'
        corr_plot = 'plt.bar(Gsingle[0:,0], Gsingle[0:,1], label=corr, alpha=0.4)'
    
    elif algorithm == 'tt2corr':
        import_corr = 'from brighteyes_ffs.fcs.atimes2corrparallel import atimes_file_2_corr as correlate'
        algorithm_settings_str = 'accuracy=resolution'
        algorithm_str = ''
        xlabel = 'Lag time (s)'
        ylabel = 'G'
        xscale = 'log'
        yscale = 'linear'
        fit_str0 = ''
        fit_str1 = 'fitresult = fcs_fit(Gexp, tau, my_fit_fun, fit_info, param, lBounds, uBounds, plotInfo=-1)'
        corr_plot = 'plt.scatter(Gsingle[1:,0], Gsingle[1:,1], s=4, label=corr)'
    
    else:
        import_corr = 'from brighteyes_ffs.fcs.fcs2corr import fcs_load_and_corr_split as correlate'
        algorithm_settings_str = 'accuracy=resolution'
        algorithm_str = ', algorithm=algorithm'
        xlabel = 'Lag time (s)'
        ylabel = 'G'
        xscale = 'log'
        yscale = 'linear'
        fit_str0 = ''
        fit_str1 = 'fitresult = fcs_fit(Gexp, tau, my_fit_fun, fit_info, param, lBounds, uBounds, plotInfo=-1)'
        corr_plot = 'plt.scatter(Gsingle[1:,0], Gsingle[1:,1], s=4, label=corr)'
    
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
        nbf.v4.new_code_cell(import_corr + """
from brighteyes_ffs.fcs.fcs_fit import fcs_fit, make_fit_info_global_fit, read_global_fit_result, stddev_2_weights
from brighteyes_ffs.fcs.get_fcs_info import get_metafile_from_file, get_file_info
from brighteyes_ffs.tools.print_tools import print_table
from brighteyes_ffs.tools.fit_curve import fit_curve
from brighteyes_ffs.fcs_gui.read_ffs import read_g_from_ffs
from brighteyes_ffs.fcs_gui.timetrace_end import timetrace_end
from brighteyes_ffs.pch.pch_fit import fit_pch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np"""),
        nbf.v4.new_markdown_cell("File name and correlation settings"),
        nbf.v4.new_code_cell("""file = \"""" + ffs_file_path + """"
list_of_g = """ + list_of_g_str +  """
list_of_g_out = """ + list_of_g_out_str +  """
averaging = """ + averaging_str +  """
algorithm = '""" + algorithm + """'
resolution = """ + resolution + """
chunksize = """ + chunksize),
        nbf.v4.new_code_cell("""mdata = get_file_info(get_metafile_from_file(file))"""),
        nbf.v4.new_markdown_cell("Calculate correlations"),
        nbf.v4.new_code_cell("""G, time_trace = correlate(file, list_of_g=list_of_g, """ + algorithm_settings_str + """, split=chunksize, time_trace=True, averaging=averaging, list_of_g_out=list_of_g_out""" + algorithm_str + """)"""),
        nbf.v4.new_markdown_cell("Plot time trace"),
        nbf.v4.new_code_cell("""num_chunks = int(np.floor(mdata.duration / chunksize))
splits = np.arange(0, (num_chunks+1)*chunksize, chunksize)
good_chunks = """ + chunks_off_str ),
        nbf.v4.new_code_cell("""n_t = timetrace_end(time_trace) # time_trace is a compressed version of the actual time_trace with between 900-1000 data points
time = np.linspace(0, mdata.duration, n_t)

fig, ax = plt.subplots(figsize=(4,2))
for i in range(np.shape(time_trace)[1]):
    plt.plot(time, time_trace[0:n_t,i])
ymin = np.min(time_trace)
ymax = np.max(time_trace)
if len(splits) <= 101:
    # color chunks red if not used for calculating average correlation
    if good_chunks is not None:
        for i in range(len(splits) - 1):
            if i not in good_chunks:
                rect = patches.Rectangle((chunksize*splits[i], ymin), chunksize*(splits[i+1]-splits[i]), ymax-ymin, fc='r', alpha=0.3)
                ax.add_patch(rect)
plt.xlim([0, mdata.duration])
plt.ylim([ymin, ymax])
plt.xlabel('Time (s)')
plt.ylabel('Photon counts per bin')"""),
        nbf.v4.new_markdown_cell("""Plot correlations  
First, calculate the average correlation of the good chunks. Then, plot the result."""),
        nbf.v4.new_code_cell("""G.average_chunks(good_chunks)"""),
        
        nbf.v4.new_code_cell("""plt.figure(figsize=(4,3))
for corr in list_of_g_out:
    Gsingle = getattr(G, corr + '_averageX')
    """ + corr_plot + """
plt.legend()
plt.xlabel('""" + xlabel + """')
plt.ylabel('""" + ylabel + """')
plt.xscale('""" + xscale + """')
plt.yscale('""" + yscale + """')"""),
]
    if analysis_object.active_fit is not None:
        # check fit
        fit_object = analysis_object.return_fit_obj()
        if fit_object is not None:
            fit_obj_single = fit_object.fit_all_curves[0]
            minbound = list_to_string(fit_obj_single.minbound, element_type="")
            maxbound = list_to_string(fit_obj_single.maxbound, element_type="")
            paramFactors10 = list_to_string(fit_obj_single.param_factors10, element_type="")
            fitfunction = fit_obj_single.fitfunction
            
            param_all_str = []
            for i_fit in range(len(fit_object.fit_all_curves)):
                fit_startv = fit_object.fit_all_curves[i_fit].startvalues
                param_all_str.append(list_to_string(fit_startv, element_type=""))
            
            if fitfunction is not None:
            
                fitfunction_label = fit_obj_single.fitfunction_label # more readable name of the fit function
                fitrange = fit_obj_single.fitrange
                fitstart = str(int(fitrange[0]))
                fitstop = str(int(fitrange[1]))
                fit_info = list_to_string(fit_obj_single.fitarray[:-1], element_type="")
                fit_startv = fit_obj_single.startvalues
                if "maximum entropy" in fit_obj_single.fitfunction_label.lower():
                    fit_startv = fit_startv[-7:]
                fitmodel = get_fit_model_from_name(fit_obj_single.fitfunction_label)
                fs = fit_startv
                fit_startv = list_to_string(fit_startv, element_type="")
                
                
                fitf = str(fitmodel.fitfunction_name)
                
                global_param = 'None'
                if 'global' in fitfunction_label:
                    global_param = list_to_string(fitmodel.global_param, element_type="")
                    rho_x = []
                    rho_y = []
                    for corr in fit_object.fit_all_curves:
                        rho_x.append(corr.startvalues[4])
                        rho_y.append(corr.startvalues[5])
                    cells += [
                    nbf.v4.new_markdown_cell("Fit correlations with global fit"),
                    nbf.v4.new_code_cell("""from """ + str(fitfunction.__module__) + """ import """ + fitfunction.__name__ + """ as my_fit_fun # you chose """ + fitfunction_label + """ as the fit function""" ),
                    nbf.v4.new_code_cell("""start_idx = """ + fitstart + """
stop_idx = """ + fitstop + """
global_param = np.asarray(""" + global_param + """)
fit_info = np.asarray(""" + fit_info + """) # use 1 for parameters that have to be fitted, 0 otherwise
lBounds = np.asarray(""" + minbound + """) # the lower bounds for the fit parameters
uBounds = np.asarray(""" + maxbound + """) # the upper bounds for the fit parameters
rho_x = """ + list_to_string(rho_x, element_type="") + """
rho_y = """ + list_to_string(rho_y, element_type="") + """
param = np.zeros((len(fit_info), len(G.list_of_g_out)))
for i, corr in enumerate(G.list_of_g_out):
    param[:, i] = np.asarray([""" + str(fs[0]) + """, """ + str(fs[1]) + """, """ + str(fs[2]) + """, """ + str(fs[3]) + """, rho_x[i], rho_y[i], """ + str(fs[6]) + """, """ + str(fs[7]) + """, """ + str(fs[8]) + """]) # starting values
G_all, tau, Gstd = G.get_av_corrs(G.list_of_g_out, '_averageX') # make 2D array with all G curves, 1D array with tau, 2D array with weights
weights = stddev_2_weights(Gstd, clipmax=1) # weights are normalized to max=1, clipmax=1 means no clipping
fitresult = fcs_fit(G_all[start_idx:stop_idx,:], tau[start_idx:stop_idx], my_fit_fun, fit_info, param, lBounds, uBounds, plotInfo=-1, global_param=global_param, weights=weights[start_idx:stop_idx])
print_table(fitresult.x)"""),
        nbf.v4.new_code_cell("""f, axs = plt.subplots(1, 3, figsize=(10,3))
for f, corr in enumerate(G.list_of_g_out):
    Gsingle = getattr(G, corr + '_averageX')
    axs[0].scatter(tau[start_idx:stop_idx], Gsingle[start_idx:stop_idx,1], s=2)
    #axs[2].scatter(tau[start_idx:stop_idx], Gsingle[start_idx:stop_idx,1], s=2)

for f, corr in enumerate(G.list_of_g_out):
    Gsingle = getattr(G, corr + '_averageX')
    axs[1].plot(tau[start_idx:stop_idx], Gsingle[start_idx:stop_idx,1]-fitresult.fun[:,f])
    axs[2].plot(tau[start_idx:stop_idx], fitresult.fun[:,f])

titles = ['G', 'Fit', 'Residuals']
for i in range(3):
    axs[i].set_xscale('log')
    axs[i].set_xlabel('Lag time (s)')
    axs[i].set_title(titles[i])"""),
                ]
                else:
                    cells += [
                    nbf.v4.new_markdown_cell("Fit correlations"),
                    nbf.v4.new_code_cell("""from """ + str(fitfunction.__module__) + """ import """ + fitfunction.__name__ + """ as my_fit_fun # you chose """ + fitfunction_label + """ as the fit function""" ),
                    nbf.v4.new_code_cell("""fitresults = []
param_all = [""" + ", ".join(param_all_str) + """]
for i, corr in enumerate(list_of_g_out):
    Gsingle = getattr(G, corr + '_averageX')
    Gexp = Gsingle[""" + fitstart + """:""" + fitstop + """,1] # your chosen range for fitting
    tau = Gsingle[""" + fitstart + """:""" + fitstop + """,0] # corresponding tau values
    fit_info = np.asarray(""" + fit_info + """) # use 1 for parameters that have to be fitted, 0 otherwise
    param = np.asarray(param_all[i])
    lBounds = np.asarray(""" + minbound + """) # the lower bounds for the fit parameters
    uBounds = np.asarray(""" + maxbound + """) # the upper bounds for the fit parameters
    """ + fit_str0 + """
    """ + fit_str1 + """
    fitresults.append(fitresult)"""),
    nbf.v4.new_code_cell("""plt.figure(figsize=(4,3))
for i, corr in enumerate(list_of_g_out):
    # plot correlation
    Gsingle = getattr(G, corr + '_averageX')
    plt.scatter(Gsingle[""" + fitstart + """:""" + fitstop + """,0], Gsingle[""" + fitstart + """:""" + fitstop + """,1], s=4, label=corr)
    # plot fit
    fitresult = fitresults[i]
    plt.plot(Gsingle[""" + fitstart + """:""" + fitstop + """,0], Gsingle[""" + fitstart + """:""" + fitstop + """,1]-fitresult.fun)
plt.legend()
plt.xlabel('""" + xlabel + """')
plt.ylabel('""" + ylabel + """')
plt.xscale('""" + xscale + """')
plt.yscale('""" + yscale + """')"""),
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
            
            elif fit_obj_single.fitfunction_label == 'Maximum entropy method free diffusion':
                cells += [
                nbf.v4.new_markdown_cell("Distribution of diffusion times"),
                nbf.v4.new_code_cell("""plt.figure(figsize=(4,3))

taumin = np.log10(Gsingle[2,0])
taumax = np.log10(Gsingle[141-1,0])
tauD = np.logspace(taumin, taumax, 200)

for i, corr in enumerate(list_of_g):
    # plot correlation
    fitresult = fitresults[i].x
    plt.plot(tauD, fitresult, label=corr)
plt.legend()
plt.xlabel('Lag time (s)')
plt.ylabel('Fraction')
plt.xscale('log')
plt.yscale('linear')"""),
            
            ]
                                     
            elif fit_object.return_all("w0") is not None and None not in fit_object.return_all("w0"):
                w0 = fit_object.return_all("w0")
                w0 = ",".join(map(str, w0))
                cells += [
                nbf.v4.new_markdown_cell("Diffusion law"),
                nbf.v4.new_code_cell("""w0 = 1e-3 * np.asarray([""" + w0 + """]) # beam waists in um
taufit = [fitresult.x[1] for fitresult in fitresults]

fitresult_difflaw = fit_curve(taufit, w0**2, 'linear', [1, 1], [1, 1], [-1e6, -1e6], [1e6, 1e6], savefig=0)

plt.figure(figsize=(3,3))
for i in range(len(taufit)):
    plt.scatter(w0[i]**2, taufit[i], edgecolors='k', marker='s')
w02fit = np.zeros(len(w0) + 1)
w02fit[0] = 0
w02fit[1:] = w0**2
taufitres = np.zeros(len(w0) + 1)
taufitres[0] = fitresult_difflaw.x[1]
taufitres[1:] = taufit - fitresult_difflaw.fun
if fitresult_difflaw.x[1] < 0:
    fitlabel = 'y = {A:.2f} x {B:.2f}'.format(A=fitresult_difflaw.x[0], B=fitresult_difflaw.x[1])
else:
    fitlabel = 'y = {A:.2f} x + {B:.2f}'.format(A=fitresult_difflaw.x[0], B=fitresult_difflaw.x[1])
plt.plot(w02fit, taufitres, '--', color='k', linewidth=0.7, label=fitlabel, zorder=1)
plt.title(fitlabel)
plt.xlabel('w0^2 (um^2)')
plt.ylabel('Diffusion time (ms)')"""),
            
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
    