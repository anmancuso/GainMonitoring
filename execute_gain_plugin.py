import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import strax
import straxen
import cutax
import plugin_Area
import spe_analysis as spe
data=os.listdir("/mnt/xedisk01/amancuso/data_calibration")
run_list=[i[0:6] for i in data if "raw_records" in i]
print(run_list)
export, __all__ = strax.exporter()
#Retrieve Data in:
st = straxen.contexts.xenonnt_online()

st.register(plugin_Area.NVLEDCalibration)
st.storage.append(strax.DataDirectory("/mnt/xedisk01/amancuso/data_calibration", readonly=False))
for i in range(len(st.storage)-1):
    st.storage.pop(0)
print(st.storage)

run_list_true=['050737']


for j in run_list_true:
    print("Processing run", j)
    try:
        os.mkdir(f"results/{j}")
        os.mkdir(f"figures/{j}")
    except:
        pass
    #try:    
    data_led=st.get_array(j,"led_cal_nv")
    data_right_cut=[]
    data_right=[]
    #gain_final=[]
    gain_final_2=[]
    pars_final=[]
    err_gain_final=[]
    score=[]
    for i in range(0, 120):
        bins=200
        arange=(-200,1200)
        channel = i
        bin_width=(arange[-1]-arange[0])/bins
        column_type=spe.which_column(channel)
        #print("channel", i, "column", column_type)
        mask_channel=data_led["channel"]==i+2000
        mask_amplitude=data_led["amplitude"]>=15
        bin_heights,bin_edges=np.histogram(data_led[mask_channel]["area"],bins=bins, range=arange)
        bin_heights_cut, _ = np.histogram(data_led[mask_channel&mask_amplitude]["area"], bins=bins, range=arange)
        bin_heights=bin_heights/bin_width
        bin_heights_cut = bin_heights_cut /bin_width
        bin_centers = bin_edges[:-1]-bin_width/2

        try:
            pars, errors, chi_square, ndof = spe.fit_spe_v2(bin_centers, bin_heights)
            pars_final.append(list(pars))
            gain,err_gain = spe.plot_functions_fit_v1(bin_centers, bin_heights, bin_heights_cut, chi_square, ndof, pars, errors,
                                             channel,j)

            #gain_final.append(spe.new_gain_calculation(bin_centers,pars))
            gain_final_2.append(gain)
            err_gain_final.append(err_gain)
            
            score.append(chi_square/ndof)
        except: 
            gain_final_2.append(0.0)
            err_gain_final.append(0.0)
            
            score.append(0.0)
    del data_led
    np.save(f"./results/{j}/gain.npy",gain_final_2)
    np.save(f"./results/{j}/pars_finals.npy",pars_final)
    np.save(f"./results/{j}/err_gain.npy",err_gain_final)
    np.save(f"./results/{j}/score.npy",score)



        #print(j,"corrupted")       

        
