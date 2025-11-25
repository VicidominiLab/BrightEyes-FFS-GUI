from brighteyes_ffs.fcs.fcs2corr import fcs_load_and_corr_split
from brighteyes_ffs.fcs.atimes_data import atimes_data_2_duration, load_atimes_data
from brighteyes_ffs.fcs.atimes2corrparallel import atimes_file_2_corr
from brighteyes_ffs.pch.atimes2pch import atimes_file_2_pch

def calc_g_wrapper(self, file, anSettings):
    if anSettings.algorithm == 'pch':
        try:
            # -------------------- PCH intensity trace --------------------
            [G, data] = fcs_load_and_corr_split(file.fname,
                                            list_of_g=anSettings.list_of_g,
                                            accuracy=30,
                                            binsize=anSettings.resolution,
                                            split=anSettings.chunksize,
                                            time_trace=True,
                                            metadata=file.metadata,
                                            root=self,
                                            list_of_g_out=anSettings.elements,
                                            algorithm=anSettings.algorithm)
        except:
            # -------------------- PCH TCPSC --------------------
            [G, data] = atimes_file_2_pch(file.fname,
                              list_of_pch=anSettings.list_of_g,
                              split=anSettings.chunksize,
                              bin_time=1e-6*anSettings.resolution,
                              normalize=True,
                              time_trace=True,
                              list_of_pch_out=anSettings.elements,
                              sysclk_MHz=240,
                              perform_calib=False,
                              max_k=30)
            
            raw_data = load_atimes_data(file.fname, channels='auto', perform_calib=False)
            raw_data.macrotime = 1e-12 # raw macrotimes must be in ps
            raw_data.microtime = 1e-12 # raw microtimes must be in ps
            file.metadata.duration = atimes_data_2_duration(raw_data, macrotime=raw_data.macrotime, subtract_start_time=False)
            
    elif anSettings.algorithm == 'tt2corr':
        # -------------------- FFS TCSPC --------------------
        [G, data] = atimes_file_2_corr(file.fname,
                                       list_of_g=anSettings.list_of_g,
                                       accuracy=anSettings.resolution,
                                       split=anSettings.chunksize,
                                       time_trace=True,
                                       root=self,
                                       list_of_g_out=anSettings.elements,
                                       averaging=anSettings.average)
        
        raw_data = load_atimes_data(file.fname, channels='auto', perform_calib=False)
        raw_data.macrotime = 1e-12 # raw macrotimes must be in ps
        raw_data.microtime = 1e-12 # raw microtimes must be in ps
        file.metadata.duration = atimes_data_2_duration(raw_data, macrotime=raw_data.macrotime, subtract_start_time=False)
        
    elif anSettings.list_of_g[0] == "crossAll":
        #check for averaging first
        # -------------------- all xcorrs FFS intensity trace --------------------
        els = anSettings.elements
        avs = anSettings.average
        if avs is not None:
            averaging = []
            for i in range(len(avs)):
                averaging.append([els[i], avs[i]])
        else:
            averaging = None
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
        # -------------------- FFS intensity trace --------------------
        [G, data] = fcs_load_and_corr_split(file.fname,
                                        list_of_g=anSettings.list_of_g,
                                        accuracy=anSettings.resolution,
                                        split=anSettings.chunksize,
                                        time_trace=True,
                                        metadata=file.metadata,
                                        root=self,
                                        list_of_g_out=anSettings.elements,
                                        algorithm=anSettings.algorithm)
    
    return G, data