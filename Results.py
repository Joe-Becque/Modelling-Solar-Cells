import numpy as np
import matplotlib.pyplot as plt
from numpy import unravel_index
import matplotlib.ticker as ticker

#file = 'C:/Users/Joe/OneDrive - Durham University/Documents/Durham/4th year/Solar cells/OO18/Optimising depths 14 (30)/Results'
file ='C:/Users/Joe/OneDrive - Durham University/Documents/Durham/4th year/Solar cells/OO22/Optimising (30)/dopingfile.txt'

len_n_vals = 30
len_p_vals = 30

d = np.genfromtxt(file, unpack=True, delimiter=' ', skip_header=1)
wl_depth = d[7].reshape((len_p_vals,len_n_vals))
ol_depth = d[8].reshape((len_p_vals,len_n_vals))
nl_depth = d[9].reshape((len_p_vals,len_n_vals))
pl_depth = d[10].reshape((len_p_vals,len_n_vals))
bl_depth = d[11].reshape((len_p_vals,len_n_vals))
N_A = d[12].reshape((len_p_vals,len_n_vals))
N_D = d[13].reshape((len_p_vals,len_n_vals))

n_start = nl_depth[0][0]
n_stop = nl_depth[-1][-1]
p_start = pl_depth[0][0]
p_stop = pl_depth[-1][-1]

czts_photogeneration = d[14].reshape((len_p_vals,len_n_vals))
czts_photogenerated_current = d[15].reshape((len_p_vals,len_n_vals))
czts_energy_absorption_efficiency 	= d[16].reshape((len_p_vals,len_n_vals))
czts_photon_absorption_efficiency 	= d[17].reshape((len_p_vals,len_n_vals))
cds_photogeneration = d[18].reshape((len_p_vals,len_n_vals))
cds_photogenerated_current = d[19].reshape((len_p_vals,len_n_vals))
cds_energy_absorption_efficiency = d[20].reshape((len_p_vals,len_n_vals))
cds_photon_absorption_efficiency = d[21].reshape((len_p_vals,len_n_vals))
total_photocurrent = d[22].reshape((len_p_vals,len_n_vals))
#pl srh t
#pl srh l
czts_total_electron_lifetime = d[23].reshape((len_p_vals,len_n_vals))
czts_current_reaching_interface = d[24].reshape((len_p_vals,len_n_vals))
czts_shunt_current = d[25].reshape((len_p_vals,len_n_vals))
#nl srh t
#nl srh l
cds_total_carrier_lifetime = d[26].reshape((len_p_vals,len_n_vals))
cds_current_reaching_interface = d[27].reshape((len_p_vals,len_n_vals))
cds_shunt_current = d[28].reshape((len_p_vals,len_n_vals))
total_shunt_current = d[29].reshape((len_p_vals,len_n_vals))
series_resistance = d[30].reshape((len_p_vals,len_n_vals))
ideal_efficiency 	= d[31].reshape((len_p_vals,len_n_vals))
ideal_max_power = d[32].reshape((len_p_vals,len_n_vals))
V_oc = d[33].reshape((len_p_vals,len_n_vals)) # (V)
J_sc = d[34].reshape((len_p_vals,len_n_vals)) # (mA cm^-2)
R_load_at_max_power = d[35].reshape((len_p_vals,len_n_vals))
fill_factor = d[36].reshape((len_p_vals,len_n_vals))
Max_Power = d[37].reshape((len_p_vals,len_n_vals))  #(W) 	
Total_Efficiency = d[38].reshape((len_p_vals,len_n_vals))

def plot_wireframe_depths(parameter, z_label, nl_depths, pl_depths):
    nl_depths, pl_depths = np.meshgrid(nl_depths, pl_depths)
    
    max_idx = unravel_index(parameter.argmax(), parameter.shape) #returns tuple (i, j)
    print('Optimised ', z_label, ' at czts depth = ', pl_depths[max_idx[1]], ' and cds depth = ', nl_depths[max_idx[0]])
    parameter = parameter.transpose()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    nl_depths = np.log(nl_depths)
    pl_depths = np.log(pl_depths)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda nl_depths, pos: ('%g') % (float(format(np.exp(nl_depths), '.1e')))))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda pl_depths, pos: ('%g') % (float(format(np.exp(pl_depths), '.1e')))))
    ax.plot_wireframe(nl_depths, pl_depths, parameter, rstride=1, cstride=1)
    ax.set_xlabel('n-layer depth (m)')
    ax.set_ylabel('p-layer depth (m)')
    ax.set_zlabel('%s' %z_label)
    plt.tight_layout()
    fig.savefig('plot_%s.pdf' %z_label)
    plt.show()
