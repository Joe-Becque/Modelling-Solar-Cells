import Material
import Spectrum
import Solarcell
import numpy as np
import matplotlib.pyplot as plt
import Constants as const
import os
import GenerateCells

'''
Generates many solar cells with different depths and doping concentrations.
Varies depths while keeping doping concentrations constant:
    CdS and CZTS depths are varied over 'length' number of values.
    This is repeated num_locs times at different CdS and CZTS doping concentrations.
Varies doping while keeping depths constant:
    CdS and CZTS doping concentrations are varied over 'length' number of values.
    This is repeated num_locs times at different CdS and CZTS depths.

Saves surface plots of selected outputs.
Creates .txt file of all outputs and determined parameters for each cell.
Creates .txt file with information of all outputs of the highest efficiency cell for each surface of doping concentrations or depths.'''

air = Material.Material('air')
window_layer = Material.Material('tio2', 'TiO2_data')
wl_depth = 100e-9
oxide_layer = Material.Material('zno', 'ZnO_data')
ol_depth = 90e-9
n_layer = Material.Material('cds', 'CdS_data')
p_layer = Material.Material('czts', 'CZTS_data')
back_layer = Material.Material('mo', 'Mo_data2')
bl_depth = 700e-9
ss = Spectrum.Spectrum('pvl')

length = 30
nl_depths = np.logspace(-9, -6, length)
pl_depths = np.logspace(-8, -5, length)
n_layer_n_d_vals = np.logspace(20, 24, length)
p_layer_n_a_vals = np.logspace(20, 24, length)

num_locs = 4
#depths to keep constant
const_pl_depths = np.logspace(-8, -5, num_locs)
const_nl_depths = np.logspace(-9, -6, num_locs)
#doping to keep constant
const_na_vals = np.logspace(20, 24, num_locs)
const_nd_vals = np.logspace(20, 24, num_locs)

maindir = os.getcwd()
print(maindir)
for i in range(len(const_na_vals)):
    for j in range(len(const_nd_vals)):
        print('p_doping[{0}], n_doping[{1}]'.format(i,j))
        workingdir = 'p_doping='+('%.4g'%const_na_vals[i])+' '+'n_doping='+('%.4g'%const_nd_vals[j])
        os.mkdir(workingdir)
        os.chdir(workingdir)
        #vary across all depths while keeping doping constant
        opt = GenerateCells.GenerateCells(ss, air,
                                window_layer, wl_depth,
                                oxide_layer, ol_depth,
                                n_layer, nl_depths,
                                p_layer, pl_depths,
                                back_layer, bl_depth,
                                n_layer_n_d_vals, p_layer_n_a_vals,
                                optimise_depths_at_n_d=const_nd_vals[j], optimise_depths_at_n_a=const_na_vals[i],
                                optimise_doping_at_nl_depth=None, optimise_doping_at_pl_depth=None,
                                opt_depths=True, opt_doping=False,
                                maindir=maindir, workingdir=os.getcwd())
        if os.getcwd() != maindir:
            os.chdir(maindir)
            
for i in range(len(const_pl_depths)):
    for j in range(len(const_nl_depths)):
        #vary across all doping while keeping depths constant
        print('pl_depth[{0}], nl_depth[{1}]'.format(i,j))
        workingdir = 'p_depth='+('%.4g'%const_pl_depths[i])+' '+'n_depth='+('%.4g'%const_nl_depths[j])
        os.mkdir(workingdir)
        os.chdir(workingdir)
        #vary across all doping while keeping depths constant
        opt = GenerateCells.GenerateCells(ss, air,
                                window_layer, wl_depth,
                                oxide_layer, ol_depth,
                                n_layer, nl_depths,
                                p_layer, pl_depths,
                                back_layer, bl_depth,
                                n_layer_n_d_vals, p_layer_n_a_vals,
                                optimise_depths_at_n_d=None, optimise_depths_at_n_a=None,
                                optimise_doping_at_nl_depth=const_nl_depths[j], optimise_doping_at_pl_depth=const_pl_depths[i],
                                opt_depths=False, opt_doping=True,
                                maindir=maindir, workingdir=os.getcwd())
        if os.getcwd() != maindir:
            os.chdir(maindir)
