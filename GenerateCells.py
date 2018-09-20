import numpy as np
import Solarcell
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from numpy import unravel_index
from matplotlib import cm
import os

class GenerateCells:
    '''Generates many solar cells with different depths and doping concentrations.
    Varies depths while keeping doping concentrations constant.
    Varies doping while keeping depths constant.'''
    def __init__(self, spectrum, air, 
                 window_layer, wl_depth, 
                 oxide_layer, ol_depth, 
                 n_layer, nl_depths, 
                 p_layer, pl_depths, 
                 back_layer, bl_depth,
                 n_layer_n_d_vals, p_layer_n_a_vals,
                 optimise_depths_at_n_d, optimise_depths_at_n_a,
                 optimise_doping_at_nl_depth, optimise_doping_at_pl_depth,
                 opt_depths=False, opt_doping=False,
                 maindir=None, workingdir=None):
        self.maindir=maindir
        self.workingdir=workingdir
        self.spectrum = spectrum
        self.air = air
        self.window_layer = window_layer
        self.oxide_layer = oxide_layer
        self.n_layer = n_layer
        self.p_layer = p_layer
        self.back_layer = back_layer
        self.wl_depth = wl_depth
        self.ol_depth = ol_depth
        self.bl_depth = bl_depth
        self.nl_depths = nl_depths
        self.pl_depths = pl_depths
        self.p_layer_n_a_vals = p_layer_n_a_vals
        self.n_layer_n_d_vals = n_layer_n_d_vals
        self.optimise_depths_at_n_d = optimise_depths_at_n_d
        self.optimise_depths_at_n_a = optimise_depths_at_n_a
        self.optimise_doping_at_nl_depth = optimise_doping_at_nl_depth
        self.optimise_doping_at_pl_depth = optimise_doping_at_pl_depth
        ###
        #OPTIMISE DEPTHS
        if opt_depths == True:
            self.optimise_depths = self.optimise_depths() #solarcells, efficiencies, optimimum_solarcell, data, data_headings
        ###
        #OPTIMISE DOPING
        if opt_doping== True:
            self.optimise_doping = self.optimise_doping() #solarcells, efficiencies, optimimum_solarcell, data, data_headings 
        
        
    def optimise_doping(self):
        n_layer_n_d_vals = self.n_layer_n_d_vals
        p_layer_n_a_vals = self.p_layer_n_a_vals
        solarcells = np.reshape([None]*len(n_layer_n_d_vals)*len(p_layer_n_a_vals), (len(n_layer_n_d_vals),len(p_layer_n_a_vals)))
        efficiencies = np.reshape([None]*len(n_layer_n_d_vals)*len(p_layer_n_a_vals), (len(n_layer_n_d_vals),len(p_layer_n_a_vals)))
        data = [[[] for x in range(len(p_layer_n_a_vals))] for x in range(len(n_layer_n_d_vals))]
        dopingfile = open('dopingfile.txt','w') 
        for i in range(len(n_layer_n_d_vals)):
            for j in range(len(p_layer_n_a_vals)):
                print('Varying doping', i,',', j)
                if os.getcwd() != self.maindir:
                    os.chdir(self.maindir)
                solarcells[i][j] = Solarcell.Solarcell(self.spectrum, self.air,
                                                        self.window_layer, self.wl_depth,
                                                        self.oxide_layer, self.ol_depth,
                                                        self.n_layer, self.optimise_doping_at_nl_depth,
                                                        self.p_layer, self.optimise_doping_at_pl_depth,
                                                        self.back_layer, self.bl_depth,
                                                        n_layer_n_d_vals[i], p_layer_n_a_vals[j])
                os.chdir(self.workingdir)
                efficiencies[i][j] = solarcells[i][j].efficiency                
                data[i][j] = list(solarcells[i][j].all_data[3].values())
                #Write headings into file
                if i==0 and j==0: 
                    for k in range(len(list(solarcells[i][j].all_data[3].values()))):
                        dopingfile.write('%s \t' % list(solarcells[i][j].all_data[3].keys())[k])
                    dopingfile.write('\n')
                #Write data into file
                for k in range(len(list(solarcells[i][j].all_data[3].values()))):
                    dopingfile.write('%s \t' % list(solarcells[i][j].all_data[3].values())[k])
                dopingfile.write('\n')
        data_headings = list(solarcells[0][0].all_data[3].keys())
        #find max efficiency
        max_efficiency_idx = unravel_index(efficiencies.argmax(), efficiencies.shape) #returns tuple (i, j)
        print('max eff idx', max_efficiency_idx)
        dopingfile.close() 
        optimimum_solarcell = solarcells[max_efficiency_idx[0]][max_efficiency_idx[1]]
        
        print('The optimised solar cell has cds doping %s and czts doping %s' %(optimimum_solarcell.n_layer_n_d, optimimum_solarcell.p_layer_n_a))
        #plots
        if os.getcwd() != self.workingdir:
            os.chdir(self.workingdir)
        self.doping_optimisation = open('doping_optimisation.txt','w')
        self.doping_optimisation.write('Max Efficiency = {0}'.format(optimimum_solarcell.efficiency))
        self.doping_optimisation.write('\n')
        self.doping_optimisation.write('n layer depth (m) = {0}'.format(self.optimise_doping_at_nl_depth))
        self.doping_optimisation.write('\n')
        self.doping_optimisation.write('p layer depth (m) = {0}'.format(self.optimise_doping_at_pl_depth))
        self.doping_optimisation.write('\n')
        self.doping_optimisation.write('n layer doping (m^-3):')
        self.doping_optimisation.write('\n')
        self.doping_optimisation.write('{0}'.format(self.n_layer_n_d_vals))
        self.doping_optimisation.write('\n')
        self.doping_optimisation.write('p layer doping (m^-3):')
        self.doping_optimisation.write('\n')
        self.doping_optimisation.write('{0}'.format(self.p_layer_n_a_vals))
        self.doping_optimisation.write('\n')
        self.plot_doping_parameter('Total Efficiency', solarcells)
        self.plot_doping_parameter('Max Power (W)', solarcells)
        self.plot_doping_parameter('J_sc (mA cm^-2)', solarcells)
        self.plot_doping_parameter('V_oc (V)', solarcells)
        self.plot_doping_parameter('total_photocurrent', solarcells)
        self.plot_doping_parameter('cds_energy_absorption_efficiency', solarcells)
        self.plot_doping_parameter('czts_energy_absorption_efficiency', solarcells)
        self.plot_doping_parameter('total_shunt_current', solarcells)
        self.plot_doping_parameter('czts_shunt_current', solarcells)
        self.plot_doping_parameter('cds_shunt_current', solarcells)
        self.doping_optimisation.write('\n')
        self.doping_optimisation.write('parameters of optimised efficiency:')
        self.doping_optimisation.write('\n')
        for i in range(len(optimimum_solarcell.all_data)):
            for key in optimimum_solarcell.all_data[i]:
                self.doping_optimisation.write('{0} \t {1}'.format(key, optimimum_solarcell.all_data[i][key]))
                self.doping_optimisation.write('\n')
            self.doping_optimisation.write('\n')
        self.doping_optimisation.close()
        optimimum_solarcell.get_j_v_curve(True, 'doping_J_V_curve.pdf') #J-V curve
        optimimum_solarcell.plot_p_layer_absorption('doping_p_layer_absorption.pdf')
        optimimum_solarcell.plot_n_layer_absorption('doping_n_layer_absorption.pdf')        
        optimimum_solarcell.plot_band_diagram('doping_band_diagram.pdf')

        return solarcells, efficiencies, optimimum_solarcell, data, data_headings    
        
    def plot_doping_parameter(self, dictkey, solarcells):
        '''Surface plot varying doping concentrations in n and p layer'''
        parameter = np.reshape([None]*len(self.n_layer_n_d_vals)*len(self.p_layer_n_a_vals), (len(self.n_layer_n_d_vals),len(self.p_layer_n_a_vals)))
        for i in range(len(self.n_layer_n_d_vals)):
            for j in range(len(self.p_layer_n_a_vals)):
                parameter[i][j] = solarcells[i][j].all_data[3]['%s' %dictkey]
        n_layer_n_d_vals, p_layer_n_a_vals = np.meshgrid(self.n_layer_n_d_vals, self.p_layer_n_a_vals)
        
        max_idx = unravel_index(parameter.argmax(), parameter.shape) #returns tuple (i, j)
        optimimum_solarcell = solarcells[max_idx[0]][max_idx[1]]
        optimised_doping_comment = 'Optimised {0} is {1} at czts doping = {2} (index {3}) and cds doping = {4} (index {5})'.format(dictkey, optimimum_solarcell, self.p_layer_n_a_vals[max_idx[1]], max_idx[1], self.n_layer_n_d_vals[max_idx[0]], max_idx[0])
        self.doping_optimisation.write(optimised_doping_comment)
        self.doping_optimisation.write('\n')
        parameter = parameter.transpose()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        n_layer_n_d_vals = np.log(n_layer_n_d_vals)
        p_layer_n_a_vals = np.log(p_layer_n_a_vals)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda n_layer_n_d_vals, pos: ('%g') % (float(format(np.exp(n_layer_n_d_vals), '.1e')))))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda p_layer_n_a_vals, pos: ('%g') % (float(format(np.exp(p_layer_n_a_vals), '.1e')))))
        ax.plot_wireframe(n_layer_n_d_vals, p_layer_n_a_vals, parameter, rstride=1, cstride=1)
        ax.set_xlabel('n_d in n layer (m^-3)')
        ax.set_ylabel('n_a in p layer (m^-3)')
        ax.set_zlabel('%s' %dictkey)
        plt.tight_layout()
        fig.savefig('plot_doping_%s.pdf' %dictkey)
        if dictkey == 'Total Efficiency':
            plt.show()
        else:
            plt.close(fig)
        return parameter
        
    def optimise_depths(self):
        nl_depths = self.nl_depths
        pl_depths = self.pl_depths
        solarcells = np.reshape([None]*len(nl_depths)*len(pl_depths), (len(nl_depths),len(pl_depths)))
        efficiencies = np.reshape([None]*len(nl_depths)*len(pl_depths), (len(nl_depths),len(pl_depths)))
        data = [[[] for x in range(len(pl_depths))] for x in range(len(nl_depths))]
        depthsfile = open('depthsfile.txt','w') 
        for i in range(len(nl_depths)):
            for j in range(len(pl_depths)):
                print('Varying depths', i,',', j)
                if os.getcwd() != self.maindir:
                    os.chdir(self.maindir)
                solarcells[i][j] = Solarcell.Solarcell(self.spectrum, self.air,
                                                        self.window_layer, self.wl_depth,
                                                        self.oxide_layer, self.ol_depth,
                                                        self.n_layer, nl_depths[i],
                                                        self.p_layer, pl_depths[j],
                                                        self.back_layer, self.bl_depth,
                                                        self.optimise_depths_at_n_d, self.optimise_depths_at_n_a)
                os.chdir(self.workingdir)
                efficiencies[i][j] = solarcells[i][j].efficiency             
                data[i][j] = list(solarcells[i][j].all_data[3].values())
                #Write headings into file
                if i==0 and j==0: 
                    for k in range(len(list(solarcells[i][j].all_data[3].values()))):
                        depthsfile.write('%s \t' % list(solarcells[i][j].all_data[3].keys())[k])
                    depthsfile.write('\n')
                #Write data into file
                for k in range(len(list(solarcells[i][j].all_data[3].values()))):
                    depthsfile.write('%s \t' % list(solarcells[i][j].all_data[3].values())[k])
                depthsfile.write('\n')
        data_headings = list(solarcells[0][0].all_data[3].keys())
        #find max efficiency
        max_efficiency_idx = unravel_index(efficiencies.argmax(), efficiencies.shape) #returns tuple (i, j)
        print('max eff idx', max_efficiency_idx)
        depthsfile.close() 
        optimimum_solarcell = solarcells[max_efficiency_idx[0]][max_efficiency_idx[1]]
        print('The optimised solar cell has cds depth %s and czts depth %s' %(optimimum_solarcell.nl_depth, optimimum_solarcell.pl_depth))
        #plots
        if os.getcwd() != self.workingdir:
            os.chdir(self.workingdir)
        self.depth_optimisation = open('depth_optimisation.txt','w')
        self.depth_optimisation.write('Max Efficiency = {0}'.format(optimimum_solarcell.efficiency))
        self.depth_optimisation.write('\n')
        self.depth_optimisation.write('n layer doping (m^-3) = {0}'.format(self.optimise_depths_at_n_d))
        self.depth_optimisation.write('\n')
        self.depth_optimisation.write('p layer doping (m^-3) = {0}'.format(self.optimise_depths_at_n_a))
        self.depth_optimisation.write('\n')
        self.depth_optimisation.write('n layer depths (m):')
        self.depth_optimisation.write('\n')
        self.depth_optimisation.write('{0}'.format(self.nl_depths))
        self.depth_optimisation.write('\n')
        self.depth_optimisation.write('p layer depths (m):')
        self.depth_optimisation.write('\n')
        self.depth_optimisation.write('{0}'.format(self.pl_depths))
        self.depth_optimisation.write('\n')
        self.plot_depth_parameter('Total Efficiency', solarcells)
        self.plot_depth_parameter('Max Power (W)', solarcells)
        self.plot_depth_parameter('J_sc (mA cm^-2)', solarcells)
        self.plot_depth_parameter('V_oc (V)', solarcells)
        self.plot_depth_parameter('total_photocurrent', solarcells)
        self.plot_depth_parameter('cds_energy_absorption_efficiency', solarcells)
        self.plot_depth_parameter('czts_energy_absorption_efficiency', solarcells)
        self.plot_depth_parameter('total_shunt_current', solarcells)
        self.plot_depth_parameter('czts_shunt_current', solarcells)
        self.plot_depth_parameter('cds_shunt_current', solarcells)
        self.depth_optimisation.write('\n')
        self.depth_optimisation.write('parameters of optimised efficiency:')
        self.depth_optimisation.write('\n')
        for i in range(len(optimimum_solarcell.all_data)):
            for key in optimimum_solarcell.all_data[i]:
                self.depth_optimisation.write('{0} \t {1}'.format(key, optimimum_solarcell.all_data[i][key]))
                self.depth_optimisation.write('\n')
            self.depth_optimisation.write('\n')
        self.depth_optimisation.close()
        optimimum_solarcell.get_j_v_curve(True, 'depths_J_V_curve.pdf') #J-V curve
        optimimum_solarcell.plot_p_layer_absorption('depths_p_layer_absorption.pdf')
        optimimum_solarcell.plot_n_layer_absorption('depths_n_layer_absorption.pdf')        
        optimimum_solarcell.plot_band_diagram('depths_band_diagram.pdf')
        return solarcells, efficiencies, optimimum_solarcell, data, data_headings

    def plot_depth_parameter(self, dictkey, solarcells):
        parameter = np.reshape([None]*len(self.nl_depths)*len(self.pl_depths), (len(self.nl_depths),len(self.pl_depths)))
        for i in range(len(self.nl_depths)):
            for j in range(len(self.pl_depths)):
                parameter[i][j] = solarcells[i][j].all_data[3]['%s' %dictkey]
        nl_depths, pl_depths = np.meshgrid(self.nl_depths, self.pl_depths)
        
        max_idx = unravel_index(parameter.argmax(), parameter.shape) #returns tuple (i, j)
        optimimum_solarcell = solarcells[max_idx[0]][max_idx[1]]
        optimised_depths_comment = 'Optimised {0} is {1} at czts depth = {2} (index {3}) and cds depth = {4} (index {5})'.format(dictkey, optimimum_solarcell, self.pl_depths[max_idx[1]], max_idx[1], self.nl_depths[max_idx[0]], max_idx[0])
        self.depth_optimisation.write(optimised_depths_comment)
        self.depth_optimisation.write('\n')
        parameter = parameter.transpose()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        nl_depths = np.log(nl_depths)
        pl_depths = np.log(pl_depths)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda nl_depths, pos: ('%g') % (float(format(np.exp(nl_depths), '.1e')))))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda pl_depths, pos: ('%g') % (float(format(np.exp(pl_depths), '.1e')))))
        ax.plot_wireframe(nl_depths, pl_depths, parameter, rstride=1, cstride=1)
        #ax.plot_surface(nl_depths, pl_depths, parameter, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        ax.set_xlabel('n-layer depth (m)')
        ax.set_ylabel('p-layer depth (m)')
        ax.set_zlabel('%s' %dictkey)
        plt.tight_layout()
        fig.savefig('plot_depths_%s.pdf' %dictkey)
        if dictkey == 'Total Efficiency':
            plt.show()
        else:
            plt.close(fig)
        return parameter
