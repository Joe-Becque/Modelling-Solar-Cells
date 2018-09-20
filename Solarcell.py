import numpy as np
from scipy.interpolate import interp1d
import Constants as const
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import Material
import Spectrum
from mpl_toolkits.mplot3d import Axes3D
from math import factorial
from collections import OrderedDict
from scipy.integrate import simps

class Solarcell:
    def __init__(self, spectrum, air,
                 window_layer, wl_depth, 
                 oxide_layer, ol_depth,
                 n_layer, nl_depth,
                 p_layer, pl_depth,
                 back_layer, bl_depth,
                 n_layer_n_d, p_layer_n_a):
        '''Model a solar cell created from layers of Material type objects and illuminated with a Spectrum.
        
        Calculates the external quantum efficiency in the n and the p layer of the solar cell.
        Calculates the photo generated current in the p and n layer.
        Models carrier lifetimes in order to calculate average diffusion lengths of carriers.
        Calculates shunt resistance and series resistance.
        Solves for the output power of a solar cell.
        '''
        self.n_layer_n_d = n_layer_n_d
        self.p_layer_n_a = p_layer_n_a
        ###
        #BUILD SOLAR CELL
        self.air = air
        self.window_layer = window_layer
        self.oxide_layer = oxide_layer
        self.n_layer = n_layer
        self.p_layer = p_layer
        self.back_layer = back_layer
        self.wl_depth = wl_depth
        self.ol_depth = ol_depth
        self.nl_depth = nl_depth
        self.pl_depth = pl_depth
        self.bl_depth = bl_depth
        self.layer_depths = np.array((self.wl_depth, self.ol_depth, self.nl_depth, self.pl_depth, self.bl_depth))
        self.spectrum = spectrum
        self.spectrum_name = self.spectrum.name
        self.reflection_at_back = 1  #assume R = 1 at the back of the pv
        temp = self.min_max_energy_index()
        self.min_energy_index = temp[0] #1931
        self.max_energy_index = temp[1] #32
        self.n_layer = self.interpolate_energy_axis(self.n_layer)
        self.p_layer =self.interpolate_energy_axis(self.p_layer)
        self.window_layer =self.interpolate_energy_axis(self.window_layer)
        self.oxide_layer =self.interpolate_energy_axis(self.oxide_layer)
        self.back_layer =self.interpolate_energy_axis(self.back_layer)
        self.spectrum = self.adjust_spectrum_data()
        self.layer_order = np.array((self.air, self.window_layer, self.oxide_layer, self.n_layer, self.p_layer, self.back_layer))
        self.layers = self.get_layer_names()
        ###
        #DOPING
        self.p_layer_doping = self.get_p_layer_doping() #n_a, n_d, E_fermi_p, n_0, p_0
        self.p_layer_n_a = self.p_layer_doping[0]
        self.p_layer_n_d = self.p_layer_doping[1]
        self.p_layer_E_fermi_p = self.p_layer_doping[2]
        self.p_layer_n_0 = self.p_layer_doping[3]
        self.p_layer_p_0 = self.p_layer_doping[4]
        self.n_layer_doping = self.get_n_layer_doping() #n_a, n_d, E_fermi_n, n_0, p_0
        self.n_layer_n_a = self.n_layer_doping[0]
        self.n_layer_n_d = self.n_layer_doping[1]
        self.n_layer_E_fermi_n = self.n_layer_doping[2]
        self.n_layer_n_0 = self.n_layer_doping[3]
        self.n_layer_p_0 = self.n_layer_doping[4]
        ###
        #QUANTUM EFFICIENCY IN P LAYER
        self.all_transmissions = self.get_all_transmissions()
        self.all_absorptions = self.get_all_absorptions()
        self.p_layer_fractions = self.get_fraction_absorbed_nth_layer(self.p_layer) #f_absorbed, f_reaching
        self.fraction_absorbed_p_layer = self.p_layer_fractions[0]
        self.fraction_reaching_p_layer = self.p_layer_fractions[1]
        self.absorption_efficiencies = self.get_absorption_efficiencies(self.fraction_absorbed_p_layer)
        self.photon_absorption_efficiency_p_layer = self.absorption_efficiencies[0]
        self.energy_absorption_efficiency_p_layer = self.absorption_efficiencies[1]
        self.generation_as_function_of_depth_p_layer = self.get_generation_as_function_of_depth(self.p_layer, self.pl_depth, self.fraction_reaching_p_layer) #generation, gen1, gen2, depths
        ###
        #QUANTUM EFFICIENCY IN N LAYER
        self.n_layer_fractions = self.get_fraction_absorbed_nth_layer(self.n_layer) #f_absorbed, f_reaching
        self.fraction_absorbed_n_layer = self.n_layer_fractions[0]
        self.fraction_reaching_n_layer = self.n_layer_fractions[1]
        self.absorption_efficiencies = self.get_absorption_efficiencies(self.fraction_absorbed_n_layer)
        self.photon_absorption_efficiency_n_layer = self.absorption_efficiencies[0]
        self.energy_absorption_efficiency_n_layer = self.absorption_efficiencies[1]
        self.generation_as_function_of_depth_n_layer = self.get_generation_as_function_of_depth(self.n_layer, self.nl_depth, self.fraction_reaching_n_layer) #generation, gen1, gen2, depths
        ###
        #PHOTOCURRENT
        self.czts_delta_n = sum(self.generation_as_function_of_depth_p_layer[0]) #(electrons per m^2)
        self.czts_delta_p = self.czts_delta_n
        self.czts_photocurrent = self.czts_delta_n * const.q
        self.cds_delta_n = sum(self.generation_as_function_of_depth_n_layer[0])
        self.cds_delta_p = self.cds_delta_n
        self.cds_photocurrent = self.cds_delta_p * const.q
        self.photocurrent = self.czts_photocurrent + self.cds_photocurrent
        ###
        #RECOMBINATION
        self.p_layerL()
        self.n_layerL()
        #RUN THESE TO CALIBRATE RECOMBINATION CONSTANTS V_SURFACE AND N_T
        #self.P_LayerFindRecombinationConstants()
        #self.N_LayerFindRecombinationConstants()
        ###
        #FERMI LEVEL SPLITTING DUE TO PHOTOCURRENT
        self.non_equilibrium_dict = self.get_non_equillibrium_details()
        ###
        #SHUNT AND SERIES RESISTANCES
        self.p_layer_shunt_info = self.get_p_layer_electron_shunt_info() #current_leaving_czts, shunt_current, prob_reaching_x0, l_av
        self.current_leaving_czts = self.p_layer_shunt_info[0] # A / m^2
        self.czts_shunt_current = self.p_layer_shunt_info[1] # A / m^2
        self.n_layer_shunt_info = self.get_n_layer_hole_shunt_info() #current_leaving_czts, shunt_current, prob_reaching_x0, l_av
        self.current_leaving_cds = self.n_layer_shunt_info[0] # A / m^2
        self.cds_shunt_current = self.n_layer_shunt_info[1] # A / m^2
        self.shunt_current = self.czts_shunt_current+ self.cds_shunt_current #A / m^2
        self.shunt_resistance_at_v_oc = const.k*const.T/const.q * np.log(self.photocurrent/self.non_equilibrium_dict['Diode saturation current density'] +1) / self.shunt_current #ohms m^2
        self.series_resistance = self.get_series_resistance() #ohms m^2
        ###
        #OUTPUTS
        self.j_v_curve = self.get_j_v_curve() #v_load, j_load, max_power, efficiency, FF, r_l_max_power, short_circuit_current, open_circuit_voltage, ideal_max_power, ideal_efficiency
        self.all_data = self.get_all_data() #cds_data, czts_data, diode_data, output_data
        self.max_power = self.j_v_curve[2]
        self.efficiency = self.j_v_curve[3]
        self.ff = self.j_v_curve[4]
        self.load_resistance_at_max_power = self.j_v_curve[5]
        self.j_sc = self.j_v_curve[6]
        self.v_oc = self.j_v_curve[7]
        self.ideal_max_power = self.j_v_curve[8]
        self.ideal_efficiency = self.j_v_curve[9]
              
    def get_all_data(self):
        '''Returns dictionaries of all input and output data'''
        cds_data = self.n_layer.setup_data
        czts_data = self.p_layer.setup_data
        diode_data = self.non_equilibrium_dict
        output_data = OrderedDict([('air',self.air.name),
                                   ('spectrum',self.spectrum.name),
                                   ('window_layer',self.window_layer.name),
                                   ('oxide_layer',self.oxide_layer.name),
                                   ('n_layer',self.n_layer.name),
                                   ('p_layer',self.p_layer.name),
                                   ('back_layer',self.back_layer.name),
                                   ('wl_depth',self.wl_depth),
                                   ('ol_depth',self.ol_depth), 
                                   ('nl_depth',self.nl_depth), 
                                   ('pl_depth',self.pl_depth), 
                                   ('bl_depth',self.bl_depth),
                                   ('N_A', self.p_layer_n_a),
                                   ('N_D', self.n_layer_n_d),
                                   ('czts_photogeneration', self.czts_delta_n),
                                   ('czts_photogenerated current', self.czts_photocurrent),
                                   ('czts_energy_absorption_efficiency', self.energy_absorption_efficiency_p_layer),
                                   ('czts_photon_absorption_efficiency', self.photon_absorption_efficiency_p_layer),
                                   ('cds_photogeneration', self.cds_delta_p),
                                   ('cds_photogenerated_current', self.cds_photocurrent),
                                   ('cds_energy_absorption_efficiency', self.energy_absorption_efficiency_n_layer),
                                   ('cds_photon_absorption_efficiency', self.photon_absorption_efficiency_n_layer),
                                   ('total_photocurrent', self.photocurrent),
                                   ('p layer n_0', self.p_layer_n_0),
                                   ('p layer p_0', self.p_layer_p_0),
                                   ('n layer n_0', self.n_layer_n_0),
                                   ('n layer p_0', self.n_layer_p_0),
                                   ('Depletion Width', self.w),
                                   ('czts_x_p', self.czts_x_p),
                                   ('cds_x_n', self.cds_x_n),
                                   ('czts_srh_electron_lifetime', self.p_layer_srh_t_e),
                                   ('czts_rad_electron_lifetime', self.p_layer_rad_t_e),
                                   ('czts_aug_electron_lifetime', self.p_layer_aug_t_e),
                                   ('czts_surf_electron_lifetime', self.p_layer_surf_t_e),
                                   ('czts_total_lifetime', self.p_layer_t_tot),
                                   ('czts_total_diffusion_length', self.p_layer_L_tot),
                                   ('czts_current_reaching_interface', self.current_leaving_czts),
                                   ('czts_shunt_current', self.czts_shunt_current),
                                   ('cds_srh_hole_lifetime', self.n_layer_srh_t_h),
                                   ('cds_rad_hole_lifetime', self.n_layer_rad_t_h),
                                   ('cds_aug_hole_lifetime', self.n_layer_aug_t_h),
                                   ('cds_surf_hole_lifetime', self.n_layer_surf_t_h),
                                   ('cds_total_lifetime', self.n_layer_t_tot),
                                   ('cds_total_diffusion_length', self.n_layer_L_tot),
                                   ('cds_current_reaching_interface', self.current_leaving_cds),
                                   ('cds_shunt_current', self.cds_shunt_current),
                                   ('total_shunt_current', self.shunt_current),
                                   ('shunt_resistance', self.shunt_resistance),
                                   ('series_resistance', self.series_resistance),
                                   ('j_0', self.j_0),
                                   ('ideal_efficiency', self.j_v_curve[9]),
                                   ('ideal_max_power', self.j_v_curve[8]),
                                   ('V_oc (V)', self.j_v_curve[7]),
                                   ('J_sc (mA cm^-2)', self.j_v_curve[6]*0.1),
                                   ('R_load_at_max_power', self.j_v_curve[5]),
                                   ('fill_factor',self.j_v_curve[4]),
                                   ('Max Power (W)',self.j_v_curve[2]),
                                   ('Total Efficiency',self.j_v_curve[3])
                                    ])
        return cds_data, czts_data, diode_data, output_data
        
    def get_j_v_curve(self, plot=False, filename='jv.pdf'):
        '''Get current density (A m^-2), voltage (V) and max power output (W m^-2) at the load 
        of the solar cell.
        
        Args:
            plot (boolean): Plot J-V curve for solar cell. Defaults to False.
            filename (str): Name of plot file.

        Returns:
            v_load (1d array): Voltage across load resistance (V).
            j_load (1d array): Current through load resistance (A m^-2).
            max_power: Maximum power output at the load resistance (W m^-2).
            efficiency: Energy efficiency of the solarcell operating at the max_power point.
            FF: Fill Factor of the solar cell.
            r_l_max_power: Load resistance required for max_power (Ohm m^2).
            short_circuit_current: Short circuit current of solar cell (A m^-2).
            open_circuit_voltage: Open circuit voltage of solar cell (V).
            ideal_max_power: Maximum power at load resistance for an ideal solar cell (W m^-2).
            ideal_efficiency: Energy efficiency of an ideal solarcell operating at the max_power point.
        '''
        r_s = self.series_resistance
        j_photocurrent = self.photocurrent
        j_shunt = 1*self.shunt_current
        j_0 = self.non_equilibrium_dict['Diode saturation current density']
        
        v_load = np.arange(-0.02,1.5,0.001)
        v_ideal_load = np.array(v_load)
        j_ideal_load = np.zeros((len(v_load)))
        ideal_power_out = np.zeros((len(v_load)))
        
        for i in range(len(v_load)):
            j_ideal_load[i] = j_photocurrent - j_0*(np.exp(const.q * v_ideal_load[i] / (const.k * const.T)) - 1)
        v_oc_ideal_idx = (np.abs(j_ideal_load)).argmin()  #index at j=0 for j ideal
        v_oc_ideal = v_ideal_load[v_oc_ideal_idx] #V_oc
        if j_shunt == 0:
            r_shunt = 1e50
        else:
            r_shunt = v_oc_ideal/j_shunt #self.non_equilibrium_dict['open circuit voltage'] / j_shunt
        if r_s == 0:
            r_s = 1e-30
        self.shunt_resistance = r_shunt
        v_load = np.array(v_load)
        j_load = np.zeros((len(v_load)))
        power_out_load = np.zeros((len(v_load)))
        a = 1 + r_s/r_shunt
        b = j_photocurrent + j_0
        c = r_shunt
        d = const.q / (const.k * const.T)
        e = r_s
        for i in range(len(v_load)):
            foo = d*e*j_0*np.exp((b*d*e/a) - (d*v_load[i]*e/(a*c)) + d*v_load[i]) / a
            bar = self.lambert_w_function(foo)
            j_load[i] = (-a*c*bar + b*c*d*e - d*e*v_load[i]) / (a*c*d*e)
        if j_load[-1] > 0:
            print('OUTPUT ERROR - v_load not measured to high enough voltage')
        
        #power output
        for i in range(len(v_load)):
            power_out_load[i] = j_load[i]*v_load[i]
            ideal_power_out[i] = j_ideal_load[i]*v_ideal_load[i]
        
        max_power = max(power_out_load) #W m^-2
        max_power_index = (np.abs(power_out_load - max_power)).argmin() #index in v_load such that v_load[i] = 0
        v_max_power = v_load[max_power_index] #V
        j_max_power = j_load[max_power_index] #A m^-2
        r_l_max_power = v_max_power/j_max_power #ohms m^2
        idx_j_sc = (np.abs(v_load)).argmin() #index in v_load such that v_load[i] = 0
        idx_v_oc = (np.abs(j_load)).argmin() #index in j_load such that j_load[i] = 0
        short_circuit_current = j_load[idx_j_sc]
        open_circuit_voltage = v_load[idx_v_oc]
        FF = max_power / (short_circuit_current * open_circuit_voltage)
        efficiency = max_power / sum(self.spectrum.irradiance)
        
        ideal_max_power = max(ideal_power_out)
        ideal_idx_v_oc = (np.abs(j_ideal_load)).argmin() #index in j_load such that j_load[i] = 0
        ideal_efficiency = ideal_max_power / sum(self.spectrum.irradiance)
        
        if plot==True:
            f, ax = plt.subplots(2, gridspec_kw = {'height_ratios':[1, 3]})
            v_load = v_load[:idx_v_oc+3]
            j_load = j_load[:idx_v_oc+3]
            j_ideal_load = j_ideal_load[:ideal_idx_v_oc+3]
            v_ideal_load = v_ideal_load[:ideal_idx_v_oc+3]
            power_out_load = power_out_load[:idx_v_oc+3]
            ideal_power_out = ideal_power_out[:ideal_idx_v_oc+3]
            ax[0].plot(v_load, power_out_load, color='blue')
            ax[0].plot(v_ideal_load, ideal_power_out, color='green')
            ax[0].set_ylabel('Power / W m$^{-2}$')
            ax[0].axhline(y=0, color='k')
            ax[0].axvline(x=0, color='k')
            ax[0].axvline(x=v_max_power, color='black', linestyle='--')
            ax[1].plot(v_load, j_load*0.1, label = 'load current', color='blue')
            ax[1].plot(v_ideal_load, j_ideal_load*0.1, color='green', label = 'ideal load current')
            ax[1].set_xlabel('Voltage / V')
            ax[1].set_ylabel('Current / mA cm$^{-2}$')#
            ax[1].axhline(y=0, color='k')
            ax[1].axvline(x=0, color='k')
            ax[1].axhline(y=short_circuit_current*0.1, color='red', linestyle='--')
            ax[1].axvline(x=open_circuit_voltage, color='red', linestyle='--')
            ax[1].axhline(y=j_max_power*0.1, color='black', linestyle='--')
            ax[1].axvline(x=v_max_power, color='black', linestyle='--')
            ax[1].legend(loc=(0.2, 0.3), fontsize='10')
            f.tight_layout()
            f.savefig(filename)
            plt.close(f)
        return v_load, j_load, max_power, efficiency, FF, r_l_max_power, short_circuit_current, open_circuit_voltage, ideal_max_power, ideal_efficiency

    def lambert_w_function(self, x):
        '''Numerical method to solve x * exp(y) - y = 0'''
        w = 0
        for i in range(1,100):
            w = w - (w*np.exp(w) - x)/(np.exp(w)*(w+1)-(w+2)*(w*np.exp(w)-x)/(2*w+2))
        return w
        
    def get_series_resistance(self):
        '''Get series resistance (Ohm m^2)'''
        ###
        series_resistance = 4.1 * 1e-4
        return series_resistance
        
    def get_p_layer_electron_shunt_info(self):
        '''Current leaving p layer (A m^-2)
           Shunt current in p layer (A m^-2)
           Probability of an electron reaching depletion region edge as function of generation depth (array)
           Mean electron diffusion length (m)'''
        l_av = self.p_layer_L_tot
        #set up depths axis
        depths = self.generation_as_function_of_depth_p_layer[3] #x_start = 0, x_stop = self.pl_depth
        gen = self.generation_as_function_of_depth_p_layer[0]
        x_p = self.non_equilibrium_dict['x_p']
        prob_reaching_x0 = np.zeros((len(depths)))
        number_reaching_x0 = np.zeros((len(depths))) 
        l_crit = x_p + l_av
        for i in range(len(depths)):
            if depths[i]<=l_crit:
                prob_reaching_x0[i] = 1
            else:
                prob_reaching_x0[i] = 0
            number_reaching_x0[i] = prob_reaching_x0[i] * gen[i]

        #sum for total number of electrons reaching interface and hence shunt current
        total_reaching_x0 = sum(number_reaching_x0)
        total_recombining = sum(gen) - total_reaching_x0
        current_leaving_czts = total_reaching_x0 * const.q
        shunt_current = total_recombining * const.q
        return current_leaving_czts, shunt_current, prob_reaching_x0, l_av
        
    def get_n_layer_hole_shunt_info(self):
        '''Returns:
        Current leaving n layer (A m^-2)
        Shunt current in n layer (A m^-2)
        Probability of a hole reaching depletion region edge as function of generation depth (array)
        Mean hole diffusion length (m)'''
        l_av = self.n_layer_L_tot
        x_n = self.non_equilibrium_dict['x_n']
        l_crit = self.nl_depth - x_n - l_av
        depths = self.generation_as_function_of_depth_n_layer[3]
        gen = self.generation_as_function_of_depth_n_layer[0]

        prob_reaching_x0 = np.zeros((len(depths)))
        number_reaching_x0 = np.zeros((len(depths)))
        for i in range(len(depths)):
            if depths[i]<l_crit:
                prob_reaching_x0[i] = 0
            else:
                prob_reaching_x0[i] = 1
            number_reaching_x0[i] = prob_reaching_x0[i] * gen[i]

        #sum for total number of holes reaching interface and hence shunt current
        total_reaching_x0 = sum(number_reaching_x0)
        total_recombining = sum(gen) - total_reaching_x0
        current_leaving_czts = total_reaching_x0 * const.q
        shunt_current = total_recombining * const.q
        if shunt_current < 1e-13:
            #small number error
            shunt_current = 0.0
        return current_leaving_czts, shunt_current, prob_reaching_x0, l_av
        
    def get_non_equillibrium_details(self):
        '''Model band alignment at p-n junction and Fermi level splitting under illumination.
           Return dictionary of diode properties.'''
        #arrange czts and cds bands relative to eachother
        #W Bao, M Ichimura data scaled to band gap of input data
        delta_E_c = 0.007583
        delta_E_v = 1.295538
        #new scaled czts valence energy
        #shift the rest of the p layer energy bands
        shift = delta_E_v   #czts_E_valence - self.p_layer.E_valence
        self.p_layer.change_energy_levels(shift)
        self.p_layer_E_fermi_p += shift

        #equillibrium picture - czts and cds share fermi level
        E_f_0 = self.n_layer_E_fermi_n
        shift = E_f_0 - self.p_layer_E_fermi_p
        if shift <= 0:
            print('CZTS FERMI LEVEL SHIFTED DOWN')
        self.p_layer.change_energy_levels(shift)
        self.p_layer_E_fermi_p += shift
        
        #built in potentials
        electron_built_in_potential = self.p_layer.E_conduction - self.n_layer.E_conduction
        hole_built_in_potential = self.p_layer.E_valence - self.n_layer.E_valence
        #p layer carrier concentrations
        p_p_0 = self.p_layer.N_valence * np.exp(- (E_f_0 - self.p_layer.E_valence)*const.q / (const.k * const.T))
        n_p_0 = self.p_layer.N_conduction * np.exp(- (self.p_layer.E_conduction - E_f_0)*const.q / (const.k * const.T))
        #n layer carrier concentrations
        p_n_0 = self.n_layer.N_valence * np.exp(- (E_f_0 - self.n_layer.E_valence)*const.q / (const.k * const.T))
        n_n_0 = self.n_layer.N_conduction * np.exp(- (self.n_layer.E_conduction - E_f_0)*const.q / (const.k * const.T))
        #new non-equilibrium picture - fermi level splitting due to photogeneration
        czts_E_fermi_p = E_f_0 - const.k * const.T * np.log((p_p_0 + self.czts_delta_p)/p_p_0)/const.q
        cds_E_fermi_n = E_f_0 + const.k * const.T * np.log((n_n_0 + self.cds_delta_n)/n_n_0)/const.q
        u_oc = const.k * const.T * np.log((n_n_0 + self.cds_delta_n)*(p_p_0 + self.czts_delta_p) / (n_n_0 * p_p_0)) / const.q  #eV
        v_oc = u_oc   #V
        
        czts_E_conduction = self.p_layer.E_conduction - u_oc
        czts_E_valence = self.p_layer.E_valence - u_oc
        if electron_built_in_potential - delta_E_c < v_oc:
            print('ERROR: V_OC BIGGER THAN BUILT IN POTENTIAL')
            v_oc = electron_built_in_potential - delta_E_c
        
        #depletion width
        czts_x_p = np.sqrt((electron_built_in_potential-delta_E_c-v_oc)*2*const.e*self.p_layer.e_r**2*self.n_layer_n_d / (const.q*(self.p_layer_n_a*self.n_layer.e_r + self.n_layer_n_d*self.p_layer.e_r)*self.p_layer_n_a))
        cds_x_n = np.sqrt((hole_built_in_potential-delta_E_v-v_oc)*2*const.e*self.n_layer.e_r**2*self.p_layer_n_a / (const.q*(self.p_layer_n_a*self.n_layer.e_r + self.n_layer_n_d*self.p_layer.e_r)*self.n_layer_n_d))
        if  czts_x_p > self.pl_depth and cds_x_n < self.nl_depth:
            czts_x_p = self.pl_depth
            cds_x_n = czts_x_p * self.p_layer_n_a * self.n_layer.e_r / (self.n_layer_n_d* self.p_layer.e_r)
        if  czts_x_p < self.pl_depth and cds_x_n > self.nl_depth:
            cds_x_n = self.nl_depth
            czts_x_p = cds_x_n * self.n_layer_n_d * self.p_layer.e_r / (self.p_layer_n_a * self.n_layer.e_r)
        if czts_x_p > self.pl_depth and cds_x_n > self.nl_depth:
            cds_x_n = self.nl_depth
            czts_x_p = cds_x_n * self.n_layer_n_d * self.p_layer.e_r / (self.p_layer_n_a * self.n_layer.e_r)
            if czts_x_p > self.pl_depth:
                czts_x_p = self.pl_depth
                cds_x_n = czts_x_p * self.p_layer_n_a * self.n_layer.e_r / (self.n_layer_n_d* self.p_layer.e_r)

        w = czts_x_p + cds_x_n
        j_0 = const.q *(self.p_layer.diffusivity_n * self.p_layer.n_i**2/(self.p_layer_L_tot * self.p_layer_n_a) + (self.n_layer.diffusivity_p * self.n_layer.n_i**2 / (self.n_layer_L_tot * self.n_layer_n_d)))
        
        self.j_0 = j_0
        self.w = w
        self.czts_x_p = czts_x_p
        self.cds_x_n = cds_x_n
        diode_data = {'E_valence_n':self.n_layer.E_valence,
                      'E_conduction_n':self.n_layer.E_conduction,
                      'E_f_0':E_f_0,
                      'E_valence_p_dark':self.p_layer.E_valence,
                      'E_conduction_p_dark':self.p_layer.E_conduction,
                      'p_p_0':p_p_0,
                      'n_p_0':n_p_0,
                      'p_n_0':p_n_0,
                      'n_n_0':n_n_0,
                      'delta_E_v':delta_E_v,
                      'delta_E_c':delta_E_c,
                      'E_valence_p_light':czts_E_valence,
                      'E_conduction_p_light':czts_E_conduction,
                      'electron_built_in_potential':electron_built_in_potential,
                      'hole_built_in_potential':hole_built_in_potential,   
                      'cds electron fermi energy':cds_E_fermi_n,
                      'czts hole fermi energy':czts_E_fermi_p,
                      'open circuit voltage':v_oc,
                      'Diode saturation current density': j_0,
                      'x_p': czts_x_p,
                      'x_n': cds_x_n,
                      'depletion width': w}
        return diode_data 
        
    def get_generation_as_function_of_depth(self, layer, layer_depth, frac_reaching_layer):
        '''Input: layer object, layer depth, fraction of light reaching the layer as function of photon energy.
        Returns:
        Generation rate G(x) as 1d array (elec. per m^2 per s)
        Generation rate G(x) as 1d array on first pass through layer (elec. per m^2 per s)
        Generation rate G(x) as 1d array on second pass through layer after reflecting (elec. per m^2 per s)
        Depths through the CZTS layer'''
        alpha = layer.absorption_coeff #absorption coefficient
        N_0 = self.spectrum.photocurrent #incident solar photocurrent
        N_p = np.zeros((len(self.spectrum.energy))) #number of photons entering layer
        frac = frac_reaching_layer
        N_p = frac * N_0
        self.d_step = layer_depth / 100
        if self.d_step > 0.1e-6:
            self.d_step = 0.1e-6
        depths = np.logspace(-10, np.log10(layer_depth), 200)# self.d_step) #array of depths through layer
         
        generation1 = np.zeros((len(self.spectrum.energy),(len(depths)))) #generation on first pass as func of photon energy and depth through layer
        transmission1 = np.zeros((len(self.spectrum.energy),(len(depths)))) #transmission on first pass as func of photon energy and depth through layer
        N_n1 = np.zeros((len(self.spectrum.energy),(len(depths)))) #fraction reaching nth depth on first pass
        generation2 = np.zeros((len(self.spectrum.energy),(len(depths)))) #generation on second pass as func of photon energy and depth through layer
        transmission2 = np.zeros((len(self.spectrum.energy),(len(depths)))) #transmission on first pass as func of photon energy and depth through layer
        N_n2 = np.zeros((len(self.spectrum.energy),(len(depths)))) #fraction reaching nth depth on second pass
        reaching_back = np.zeros((len(self.spectrum.energy))) #fraction reaching back of layer
        reflected = np.zeros((len(self.spectrum.energy))) #fraction reflected at back of layer
        #find next layer
        for i in range(len(self.layers)):
            if self.layers[i] == layer.name:
                next_layer = self.layer_order[i+1]
        reflectance = self.get_reflectance(layer, next_layer) # reflection coefficients at back of layer as func of photon energy
        #find G(x, E)
        for j in range(len(depths)):
            for i in range(len(self.spectrum.energy)):
                var = np.exp(- alpha[i] * (depths[j]-depths[j-1]))
                if j == 0:
                    #values at surface
                    N_n1[i][j] = N_p[i] 
                    transmission1[i][j] = N_n1[i][j]
                    reaching_back[i] = N_p[i] * np.exp(- alpha[i] * layer_depth) #G = N_0 (1 - exp(- alpha delta_x))
                    #reflection
                    reflected[i] = reaching_back[i] * reflectance[i]
                    #values at back
                    N_n2[i][j] = reflected[i]
                    transmission2[i][j] = N_n2[i][j]
                else:
                    #first pass values at depth j 
                    N_n1[i][j] = transmission1[i][j-1]
                    transmission1[i][j] = N_n1[i][j] * var
                    generation1[i][j] += N_n1[i][j] * (1-var)
                    #second pass values at depth (layer_depth + j)
                    N_n2[i][j] = transmission2[i][j-1]
                    transmission2[i][j] = N_n2[i][j] * var
                    generation2[i][j] = N_n2[i][j] * (1-var)
        #transpose so generations can be summed over photon energies
        generation2 = generation2[:,::-1].transpose() #reverse so x axis aligns with generation1
        generation1 = generation1.transpose()
        gen1 = np.zeros((len(depths)))
        gen2 = np.zeros((len(depths)))
        #get generations as function of depth summed over all wavelengths
        for i in range(len(depths)):
            gen1[i] = sum(generation1[i])
            gen2[i] = sum(generation2[i])
        generation = gen1 + gen2 #function of depth only
        return generation, gen1, gen2, depths

    def get_absorption_efficiencies(self, frac_absorbed):
        '''Input 1d array fraction of light absorbed in a layer as function of photon energy.
        Returns:
        Fraction of incident photons absorbed in the p-doped layer.
        Fraction of incident energy absorbed in the p-doped layer.'''
        total_photons_absorbed = 0
        total_energy_absorbed = 0
        #find (faction at energy) * (current at energy)
        for i in range(len(self.n_layer.absorption_coeff)):
            #find (faction at energy) * (photocurrent at energy)
            total_photons_absorbed += frac_absorbed[i] * self.spectrum.photocurrent[i]
            #find (faction at energy) * (irradiance at energy)
            total_energy_absorbed += frac_absorbed[i] * self.spectrum.irradiance[i]
        #find efficiencies
        photon_absorption_efficiency = total_photons_absorbed / sum(self.spectrum.photocurrent)
        energy_absorption_efficiency = total_energy_absorbed / sum(self.spectrum.irradiance)
        return photon_absorption_efficiency, energy_absorption_efficiency
        
    def get_fraction_absorbed_nth_layer(self, material):
        '''Returns: 
        1d array of the fraction of light absorbed in nth layer as a function of photon energy.
        1d array of the fraction of light reaching the nth layer as a function of photon energy.'''
        T = np.zeros((len(self.layer_order), len(self.spectrum.energy))) #fraction LEAVING layer
        R = np.zeros((len(self.layer_order)+1, len(self.spectrum.energy)))
        frac_absorbed = np.zeros((len(self.layer_order), len(self.spectrum.energy)))
        frac_transmitted = np.zeros((len(self.layer_order), len(self.spectrum.energy)))
        T[len(self.layer_order)-1] = 1 # self.reflection_at_back #transmission at back of cell
        frac_transmitted[0] = 1 #absorption through air accounted for in spectrum
        frac_absorbed[0] = 0 #absorption through air accounted for in spectrum
        all_transmissions = self.all_transmissions
        all_absorptions = self.all_absorptions
        for i in range(1, len(self.layer_order)):
            if self.layer_order[i] == material:
                material_index = i
            T_R = self.get_T_R(self.layer_order[i-1], self.layer_order[i])
            T[i-1] = T_R[0] #1d list (energy)
            R[i-1] = T_R[1]
            frac_transmitted[i] = all_transmissions[i-1]
            frac_absorbed[i] = all_absorptions[i-1]
        f_reaching = np.ones((len(self.spectrum.energy)))
        f_first_pass = np.ones((len(self.spectrum.energy)))
        f_second_pass = np.ones((len(self.spectrum.energy)))
        for j in range(len(self.spectrum.energy)):
            for i in range(material_index):
                f_reaching[j] *= frac_transmitted[i][j] * T[i][j]
            f_first_pass[j] = f_reaching[j] * frac_absorbed[material_index][j]
            f_second_pass[j] = f_reaching[j] * frac_transmitted[material_index][j] * R[material_index][j] * frac_absorbed[material_index][j]
        f_total = f_first_pass + f_second_pass
        return f_total, f_reaching
        
    def get_fraction_absorbed_p_layer(self):
        '''Returns 1d array of fractions absorbed in the p doped layer as a function of photon energy'''
        #create empty arrays with dimensions of (number of layers) and (length of spectrum energy array)
        T = np.zeros((len(self.layer_order), len(self.spectrum.energy))) #fraction LEAVING layer at interface
        R = np.zeros((len(self.layer_order)+1, len(self.spectrum.energy))) #fraction reflected at interface
        frac_absorbed = np.zeros((len(self.layer_order), len(self.spectrum.energy))) #frac absorbed in bulk
        frac_transmitted = np.zeros((len(self.layer_order), len(self.spectrum.energy))) #frac transmitted through bulk
        T[len(self.layer_order)-1] = 1 #transmission at back of cell
        frac_transmitted[0] = 1        #absorption through air accounted for in spectrum
        frac_absorbed[0] = 0           #absorption through air accounted for in spectrum
        all_transmissions = self.all_transmissions
        all_absorptions = self.all_absorptions
        for i in range(1, len(self.layer_order)): #start from 1 to exclude air layer
            if self.layer_order[i] == self.p_layer: #find position of p layer
                p_layer_index = i
            T_R = self.get_T_R(self.layer_order[i-1], self.layer_order[i])
            T[i-1] = T_R[0] # T[i-1] = 1d array (as function of energy) of transmissions at interface i-1
            R[i-1] = T_R[1]
            frac_transmitted[i] = all_transmissions[i-1]
            frac_absorbed[i] = all_absorptions[i-1]
        f_reaching = np.ones((len(self.spectrum.energy)))
        f_first_pass = np.ones((len(self.spectrum.energy)))
        f_second_pass = np.ones((len(self.spectrum.energy)))
        
        for j in range(len(self.spectrum.energy)):
            for i in range(p_layer_index):
                f_reaching[j] = frac_transmitted[1][j]*frac_transmitted[2][j]*frac_transmitted[3][j]*T[0][j]*T[1][j]*T[2][j]*T[3][j]
            f_first_pass[j] = f_reaching[j] * frac_absorbed[p_layer_index][j]
            f_second_pass[j] = f_reaching[j] * frac_transmitted[p_layer_index][j] * R[p_layer_index][j] * frac_absorbed[p_layer_index][j]
        f_total = f_first_pass + f_second_pass
        return f_total, f_reaching, T, frac_transmitted
    
    def get_all_transmissions(self):
        '''Returns 1d array of transmission coefficients as a function of photon energy for each material'''
        length = len(self.window_layer.absorption_coeff)
        wl_transmission = np.zeros((length))
        ol_transmission = np.zeros((length))
        nl_transmission = np.zeros((length))
        pl_transmission = np.zeros((length))
        bl_transmission = np.zeros((length))
        for i in range(length):
            wl_transmission[i] = np.exp(- self.window_layer.absorption_coeff[i] * self.wl_depth)
            ol_transmission[i] = np.exp(- self.oxide_layer.absorption_coeff[i] * self.ol_depth)
            nl_transmission[i] = np.exp(- self.n_layer.absorption_coeff[i] * self.nl_depth)
            pl_transmission[i] = np.exp(- self.p_layer.absorption_coeff[i] * self.pl_depth)
            bl_transmission[i] = np.exp(- self.back_layer.absorption_coeff[i] * self.bl_depth)
        return wl_transmission, ol_transmission, nl_transmission, pl_transmission, bl_transmission
    
    def get_all_absorptions(self):
        '''Returns 1d array of absorption coefficients as a function of photon energy for each material'''
        length = len(self.window_layer.absorption_coeff)
        wl_absorption = np.zeros((length))
        ol_absorption = np.zeros((length))
        nl_absorption = np.zeros((length))
        pl_absorption = np.zeros((length))
        bl_absorption = np.zeros((length))
        for i in range(length):
            wl_absorption[i] = 1 - np.exp(- self.window_layer.absorption_coeff[i] * self.wl_depth)
            ol_absorption[i] = 1 - np.exp(- self.oxide_layer.absorption_coeff[i] * self.ol_depth)
            nl_absorption[i] = 1 - np.exp(- self.n_layer.absorption_coeff[i] * self.nl_depth)
            pl_absorption[i] = 1 - np.exp(- self.p_layer.absorption_coeff[i] * self.pl_depth)
            bl_absorption[i] = 1 - np.exp(- self.back_layer.absorption_coeff[i] * self.bl_depth)
        return wl_absorption, ol_absorption, nl_absorption, pl_absorption, bl_absorption    
    
    def get_absorption(self, material):
        '''Returns 1d array of absorption coefficient for a material as a function of photon energy'''
        absorption = np.zeros((len(material.absorption_coeff))) #1d array for absorption as function of photon energy
        for i in range(len(material.absorption_coeff)):
            absorption[i] = 1 - np.exp(- material.absorption_coeff[i] * material.depth)
        return absorption
        
    def get_reflectance(self, material1, material2, theta_i=const.theta_i):
        '''Fraction reflected at an interface from Fresnel's equations'''        
        n_1 = material1.refractive_index
        n_2 = material2.refractive_index
        if type(n_1)==int or type(n_1)==float:
            n_1 = [n_1 for i in range(len(n_2))]
        R_par, R_perp, R = np.zeros((len(n_1))), np.zeros((len(n_1))), np.zeros((len(n_1)))
        for i in range(len(n_1)):
            R_par[i] = ((n_1[i]*(1 - (n_1[i] * np.sin(theta_i) / n_2[i])**2)**0.5 - n_2[i] * np.cos(theta_i)) / (n_1[i]*(1 - (n_1[i] * np.sin(theta_i) / n_2[i])**2)**0.5 + n_2[i] * np.cos(theta_i)))**2
            R_perp[i] = ((n_1[i]*np.cos(theta_i) - n_2[i]*(1 - (n_1[i]*np.sin(theta_i) / n_2[i])**2)**0.5) / ((n_1[i] * np.cos(theta_i) + n_2[i]*(1 - (n_1[i] * np.sin(theta_i) / n_2[i])**2)**0.5)))**2
            R[i] = 0.5*(R_par[i] + R_perp[i])           #assume light is unpolarised
        return R
    
    def get_T_R(self, material1, material2, theta_i=const.theta_i):
        '''Fraction of light transmitted and reflected at an interface as function of photon energy
        using Fresnel's equations'''        
        n_1 = material1.refractive_index #1d array of refractive index as func of photon energy
        n_2 = material2.refractive_index #1d array of refractive index as func of photon energy
        if type(n_1)==int or type(n_1)==float:
            n_1 = [n_1 for i in range(len(n_2))]
        R_par, R_perp, R, T = np.zeros((len(n_1))), np.zeros((len(n_1))), np.zeros((len(n_1))), np.zeros((len(n_1)))
        for i in range(len(n_1)):
            R_par[i] = ((n_1[i]*(1 - (n_1[i] * np.sin(theta_i) / n_2[i])**2)**0.5 - n_2[i] * np.cos(theta_i)) / (n_1[i]*(1 - (n_1[i] * np.sin(theta_i) / n_2[i])**2)**0.5 + n_2[i] * np.cos(theta_i)))**2
            R_perp[i] = ((n_1[i]*np.cos(theta_i) - n_2[i]*(1 - (n_2[i]*np.sin(theta_i) / n_2[i])**2)**0.5) / ((n_1[i] * np.cos(theta_i) + n_2[i]*(1 - (n_1[i] * np.sin(theta_i) / n_2[i])**2)**0.5)))**2
            R[i] = 0.5*(R_par[i] + R_perp[i])           #assume light is unpolarised
            T[i] = 1 - R[i]
        return T, R
        
    def min_max_energy_index(self):
        '''Returns the index of:
            first spectrum energy larger than the highest material energy at first index
            last spectrum energy smaller than the smallest material energy at last index'''
        min_index = -1
        max_index = -1
        highest_energy_start = max(self.n_layer.energy[0], self.p_layer.energy[0], self.window_layer.energy[0], self.oxide_layer.energy[0], self.back_layer.energy[0])
        lowest_energy_end = min(self.n_layer.energy[-1], self.p_layer.energy[-1], self.window_layer.energy[-1], self.oxide_layer.energy[-1], self.back_layer.energy[-1])
        for i in range(len(self.spectrum.energy)):
            if self.spectrum.energy[i] > highest_energy_start:
                if i>min_index:
                    min_index = i
                break
        for j in range(1,len(self.spectrum.energy)):
            if self.spectrum.energy[-j] < lowest_energy_end:
                if j>max_index:
                    max_index = j 
                break
        return min_index, max_index
            
    def interpolate_energy_axis(self, material):
        '''Interpolate solar spectrum so the x-axis matches the material's properties x-axis'''
        new_material = Material.Material(material.name, material.data_file)
        data = material.data
        data_new = []
        min_index = self.min_energy_index
        max_index = self.max_energy_index
        spectrum_energy_cropped = self.spectrum.energy[min_index:-max_index]
        data_new.append(spectrum_energy_cropped)
        for k in range(1, len(data)):
            f = interp1d(material.energy, data[k])
            data_k_new = f(spectrum_energy_cropped)
            data_new.append(data_k_new)
        new_material.change_data(data_new)
        return new_material

    def adjust_spectrum_data(self):
        '''Crops spectrum data so that the energies lie within the range of data available
        from all the materials involved'''
        new_spectrum = Spectrum.Spectrum(self.spectrum_name)
        min_index = self.min_energy_index
        max_index = self.max_energy_index
        new_spectrum.energy = self.spectrum.energy[min_index:-max_index]
        new_spectrum.irradiance = self.spectrum.irradiance[min_index:-max_index]
        new_spectrum.energy_joules = self.spectrum.energy_joules[min_index:-max_index]
        new_spectrum.photocurrent = self.spectrum.photocurrent[min_index:-max_index]
        new_spectrum.wavelength = self.spectrum.wavelength[min_index:-max_index]
        return new_spectrum

    def get_layer_names(self):
        '''Returns list of names of layers in the cell'''
        layer_order = self.layer_order
        names = [None]*len(layer_order)
        for i in range(len(layer_order)):
            names[i] = layer_order[i].name
        return names
            
    def p_layerL(self):
        kt = const.k * const.T
        n = self.p_layer_n_0 + self.czts_delta_n
        p = self.p_layer_p_0 + self.czts_delta_p
        bar = self.p_layer.E_g / (520.57-143.76)
        #Energy levels
        E_v_cu = (150-143.76) * bar + self.p_layer.E_valence
        E_v_zn = (199-143.76) * bar + self.p_layer.E_valence
        E_v_sn1 = (229-143.76) * bar + self.p_layer.E_valence
        E_v_sn2 = (240-143.76)* bar + self.p_layer.E_valence
        E_v_sn3 = (260-143.76)* bar + self.p_layer.E_valence
        E_cu_zn = (174-143.76)* bar + self.p_layer.E_valence
        E_cu_sn1 = (218-143.76)* bar + self.p_layer.E_valence
        E_cu_sn2 = (250-143.76)* bar + self.p_layer.E_valence
        E_cu_sn3 = (283-143.76)* bar + self.p_layer.E_valence
        E_zn_sn = (204-143.76)* bar + self.p_layer.E_valence
        E_sn_cu1 = (331-143.76)* bar + self.p_layer.E_valence
        E_sn_cu2 = (393-143.76)* bar + self.p_layer.E_valence
        E_sn_cu3 = (519-143.76)* bar + self.p_layer.E_valence
        E_zn_cu = (483-143.76)* bar + self.p_layer.E_valence
        E_sn_zn1 = (480-143.76)* bar + self.p_layer.E_valence
        E_sn_zn2 = (472-143.76)* bar + self.p_layer.E_valence
        E_cu_i = (483-143.76)* bar + self.p_layer.E_valence
        E_zn_i = (472-143.76)* bar + self.p_layer.E_valence
        E_sn_i1 = (488-143.76)* bar + self.p_layer.E_valence
        E_sn_i2 = (346-143.76)* bar + self.p_layer.E_valence
        E_sn_i3 = (333-143.76)* bar + self.p_layer.E_valence
        E_v_s = (309-143.76)* bar + self.p_layer.E_valence
        xs = [0,1,2,2,2,3,4,4,4,5,6,6,6,7,8,8,9,10,11,11,11,12]
        self.state_level = [E_v_cu, E_v_zn, E_v_sn1,E_v_sn2,E_v_sn3, 
        E_cu_zn, E_cu_sn1,E_cu_sn2,E_cu_sn3, E_zn_sn, E_sn_cu1,E_sn_cu2,E_sn_cu3,
        E_zn_cu, E_sn_zn1,E_sn_zn2, E_cu_i, E_zn_i, E_sn_i1,E_sn_i2,E_sn_i3, E_v_s]
        #formation energies
        VCu = 0.77
        VZn = 1.12
        VSn1 = 2.82
        VSn2 = 2.82
        VSn3 = 2.82
        CuZn = 0.01
        CuSn1 = 0.87
        CuSn2 = 0.87
        CuSn3 = 0.87
        ZnSn = 0.69
        SnCu1 = 6.54
        SnCu2 = 6.54
        SnCu3 = 6.54
        ZnCu = 2.42
        SnZn1 = 4.11
        SnZn2 = 4.11
        Cui = 3.13 
        Zni = 5.92 
        Sni1 = 8.11
        Sni2 = 8.11
        Sni3 = 8.11
        Vs = 0.99
        formation_energies = [VCu, VZn, VSn1,VSn2,VSn3, CuZn, CuSn1,CuSn2,CuSn3, ZnSn, SnCu1,SnCu2,SnCu3, ZnCu, SnZn1,SnZn2, Cui, Zni, Sni1,Sni2,Sni3, Vs]
        construction_temp = 400 + 279.8 #Kelvin
        construction_energy = const.k*construction_temp / const.q #eV
        Z = 0
        for i in range(len(formation_energies)):
            Z += np.exp(-formation_energies[i]/ construction_energy )
        boltzmann_probabilities = np.zeros((len(formation_energies)))
        for i in range(len(formation_energies)):
            boltzmann_probabilities[i] = np.exp(- formation_energies[i] / construction_energy )/Z
        N_ts_boltzmann = boltzmann_probabilities*self.p_layer.N_T
        t_h_mins = np.zeros((len(xs)))
        t_e_mins = np.zeros((len(xs)))
        R = np.zeros((len(xs)))
        t = np.zeros((len(xs)))
        for i in range(len(xs)):
            t_h_mins[i] = 1 / (self.p_layer.sigma_h * self.p_layer.v_th * N_ts_boltzmann[i])
            t_e_mins[i] = 1 / (self.p_layer.sigma_e * self.p_layer.v_th * N_ts_boltzmann[i])
            R[i] =(n*p - self.p_layer.n_i**2) / (t_h_mins[i]*(n + self.p_layer.N_conduction*np.exp(- (self.p_layer.E_conduction-self.state_level[i])*const.q/kt)) + t_e_mins[i]*(p+self.p_layer.N_valence*np.exp(- (self.state_level[i]-self.p_layer.E_valence)*const.q/kt)))
            t[i] = self.czts_delta_n / R[i]
        t_tot = 0
        for i in range(len(t)):
            t_tot += 1/t[i]
        
        self.p_layer_srh_t_e = 1/t_tot
        self.p_layer_surf_t_e = self.pl_depth/(2*self.p_layer.v_surface)
        self.p_layer_rad_t_e = 1 / (self.p_layer.c_rad_n * self.p_layer_p_0)
        self.p_layer_aug_t_e = 1  /(self.p_layer.c_aug_n * self.p_layer_p_0**2)
        self.R_surf = self.czts_delta_n / self.p_layer_surf_t_e
        self.R_srh = sum(R)
        self.p_layer_t_tot = 1 /(1/self.p_layer_srh_t_e + 1/self.p_layer_surf_t_e + 1/self.p_layer_rad_t_e + 1/self.p_layer_aug_t_e)
        self.p_layer_L_tot = np.sqrt(self.p_layer_t_tot * self.p_layer.diffusivity_n)

    
    def n_layerL(self):
        '''srh and surface recomb'''
        kt = const.k * const.T
        n_i = self.n_layer.n_i
        n = self.n_layer_n_0 + self.cds_delta_n
        p = self.n_layer_p_0 + self.cds_delta_p
        E_t = self.n_layer.E_g*0.5 + self.n_layer.E_valence
        N_t = self.n_layer.N_T              #density of trapped states
        t_e_min = (1 / (self.n_layer.sigma_e * self.n_layer.v_th * N_t))
        t_h_min = (1 / (self.n_layer.sigma_h * self.n_layer.v_th * N_t))
        R_test = (n*p - n_i**2) / (t_h_min*(n + self.n_layer.N_conduction*np.exp(- (self.n_layer.E_conduction-E_t)*const.q/kt)) + t_e_min*(p+self.n_layer.N_valence*np.exp(- (E_t-self.n_layer.E_valence)*const.q/kt)))
        self.n_layer_srh_t_h = self.cds_delta_n / R_test
        self.n_layer_surf_t_h = self.nl_depth / (2 * self.n_layer.v_surface)
        self.n_layer_rad_t_h = 1 / (self.n_layer.c_rad_p * self.n_layer_n_0)
        self.n_layer_aug_t_h = 1  /(self.n_layer.c_aug_p * self.n_layer_n_0**2)
        self.n_layer_t_tot = 1 /(1/self.n_layer_srh_t_h + 1/self.n_layer_surf_t_h + 1/self.n_layer_rad_t_h + 1/self.n_layer_aug_t_h) 
        self.n_layer_L_tot = np.sqrt(self.n_layer_t_tot * self.n_layer.diffusivity_p)

    def get_p_layer_doping(self):
        '''Returns p layer density of acceptors (m^-3), density of donors (m^-3),
        quasi fermi level (eV), and density of electrons and holes in equillibrium (m^-3)'''
        n_a = self.p_layer_n_a             #density of acceptors (m^-3)
        n_d = 1                            #density of donors (m^-3)
        E_fermi_p = self.p_layer.E_valence + const.k*const.T/const.q*np.log(self.p_layer.N_valence / n_a)      #doped fermi energy equilibrium
        n_0 = self.p_layer.n_i**2 / n_a    #equilibrium doped n conc. (m^-3)
        p_0 = n_a                          #equilibrium doped p conc. (m^-3)
        if self.p_layer.N_conduction < n_d:
            print('n doping exceeds DOS')
            self.n_d = self.N_conduction
        if self.p_layer.N_valence < n_a:
            print('p doping exceeds DOS')
            n_a = self.p_layer.N_valence
        return n_a, n_d, E_fermi_p, n_0, p_0
        
    def get_n_layer_doping(self):
        '''Returns n layer density of acceptors (m^-3), density of donors (m^-3), 
        quasi fermi level (eV), and density of electrons and holes in equillibrium (m^-3)'''
        n_a = 1                            #density of acceptors (m^-3)
        n_d = self.n_layer_n_d # 1e20  #   #density of donors (m^-3)
        E_fermi_n = self.n_layer.E_conduction - const.k*const.T/const.q*np.log(self.n_layer.N_conduction / n_d) #quasi fermi energy
        n_0 = n_d                          #equilibrium doped n conc. (m^-3)
        p_0 = self.n_layer.n_i**2 / n_d    #equilibrium doped p conc. (m^-3)
        if self.n_layer.N_conduction < n_d:
            print('n doping exceeds DOS')
            n_d = self.n_layer.N_conduction
        if self.n_layer.N_valence < n_a:
            print('p doping exceeds DOS')
            n_a = self.n_layer.N_valence
        return n_a, n_d, E_fermi_n, n_0, p_0
    
    def plot_p_layer_absorption(self, filename):
        '''plots Generation (m^-3 s^-1) and probability of reaching pn interface in the
        p layer as a function of depth (m)'''
        gen = self.generation_as_function_of_depth_p_layer[0]
        depths = self.generation_as_function_of_depth_p_layer[3]
        prob_reaching_x0 = self.p_layer_shunt_info[2]
        l_av = self.p_layer_shunt_info[3]
        x_p = self.non_equilibrium_dict['x_p']
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(depths*1e6, gen, color='blue', label='Generation Rate')
        ax1.set_xlabel('Distance through CZTS, $x$ ($\mu$m)')
        ax1.set_ylabel('Generation of electrons (electrons / s)')
        ax2.plot(depths*1e6, prob_reaching_x0, color='green', label='Probability of Reaching Interface')
        ax2.set_ylabel('Probability of reaching interface')
        if x_p>0 and l_av>0:
            ax2.axvline(x=x_p*1e6, color='red', linestyle='--', label='Depletion Edge')
            ax2.axvline(x=l_av*1e6, color='black', linestyle='--', label='Recombination Distance')
        ax2.set_ylim([0,1.1])
        plt.tight_layout()
        #plt.show()
        fig.savefig(filename)
        plt.close(fig)
    
    def plot_n_layer_absorption(self, filename):
        '''plots Generation (m^-3 s^-1) and probability of reaching pn interface in the
        n layer as a function of depth (m)'''
        gen = self.generation_as_function_of_depth_n_layer[0]
        depths = self.generation_as_function_of_depth_n_layer[3]
        prob_reaching_x0 = self.n_layer_shunt_info[2]
        l_av = self.n_layer_shunt_info[3]
        x_n = self.nl_depth - self.non_equilibrium_dict['x_n']
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(depths*1e6, gen, color='blue', label='Generation Rate')
        ax1.set_xlabel('Distance through CdS, $x$ ($\mu$m)')
        ax1.set_ylabel('Generation of electrons (electrons / s)')
        ax2.plot(depths*1e6, prob_reaching_x0, color='green', label='Probability of Reaching Interface')
        ax2.set_ylabel('Probability of reaching interface')
        ax2.axvline(x=x_n*1e6, color='red', linestyle='--', label='Depletion Edge')
        ax2.axvline(x=l_av*1e6, color='black', linestyle='--', label='Recombination Distance')
        ax2.set_ylim([0,1.1])
        #plt.legend(loc=(0.3,0.84), fontsize='10')
        plt.tight_layout()
        #plt.show()
        fig.savefig(filename)
        plt.close(fig)
        
    def plot_band_diagram(self, filename):
        '''Plot energy band diagram across pn junction'''
        values = self.non_equilibrium_dict
        #vertical lines
        x_n = values['x_n']
        x_p = values['x_p']
        x_cds = self.nl_depth
        x_czts = self.pl_depth
        #horizontal lines
        delta_E_v = values['delta_E_v']
        delta_E_c = values['delta_E_c']
        cds_valence = values['E_valence_n'] #0
        xs_cds_valence = [-x_cds, -x_n]
        ys_cds_valence = [cds_valence,cds_valence]
        cds_conduction = values['E_conduction_n'] #2.3
        xs_cds_conduction = [-x_cds, -x_n]
        ys_cds_conduction = [cds_conduction,cds_conduction]
        czts_valence = values['E_valence_p_light'] #2.75
        xs_czts_valence = [x_p, x_czts]
        ys_czts_valence = [czts_valence,czts_valence]
        czts_conduction = values['E_conduction_p_light']
        xs_czts_conduction = [x_p, x_czts]
        ys_czts_conduction = [czts_conduction,czts_conduction]
        cds_fermi_n = values['cds electron fermi energy']
        xs_cds_fermi_n = [-x_cds, x_p]
        ys_cds_fermi_n = [cds_fermi_n,cds_fermi_n]
        czts_fermi_p = values['czts hole fermi energy']
        xs_czts_fermi_p = [-x_n, x_czts]
        ys_czts_fermi_p = [czts_fermi_p,czts_fermi_p]
        #calculate depletion region band gaps cds
        xs_cds_depletion = np.arange(-x_n, 0, x_n/50)
        cds_valence_depletion = np.zeros((len(xs_cds_depletion)))
        cds_conduction_depletion = np.zeros((len(xs_cds_depletion)))
        for i in range(len(xs_cds_depletion)):
            cds_valence_depletion[i] = cds_valence + const.q*self.n_layer_n_d/(2*self.n_layer.e)*(xs_cds_depletion[i] + x_n)**2
            cds_conduction_depletion[i] = cds_conduction + const.q*self.n_layer_n_d/(2*self.n_layer.e)*(xs_cds_depletion[i] + x_n)**2
        #calculate depletion region band gaps czts
        xs_czts_depletion = np.arange(0, x_p, x_p/50)
        czts_valence_depletion = np.zeros((len(xs_czts_depletion)))
        czts_conduction_depletion = np.zeros((len(xs_czts_depletion)))
        for i in range(len(xs_czts_depletion)):
            czts_valence_depletion[i] = const.q*self.p_layer_n_a*(x_p*xs_czts_depletion[i] - 0.5*xs_czts_depletion[i]**2)/self.p_layer.e + const.q*self.n_layer_n_d*x_n**2/self.n_layer.e + delta_E_v
            czts_conduction_depletion[i] = cds_conduction + const.q*self.p_layer_n_a*(x_p*xs_czts_depletion[i] - 0.5*xs_czts_depletion[i]**2)/self.p_layer.e + const.q*self.n_layer_n_d*x_n**2/self.n_layer.e + delta_E_c
        xs_depletion = np.concatenate((xs_cds_depletion,xs_czts_depletion))
        ys_valence = np.concatenate((cds_valence_depletion,czts_valence_depletion))
        ys_conduction = np.concatenate((cds_conduction_depletion,czts_conduction_depletion))
        #plot
        ax = plt.subplot(111)
        ax.plot(xs_depletion, ys_valence, color='red')
        ax.plot(xs_czts_valence,ys_czts_valence, color='red')
        ax.plot(xs_cds_valence,ys_cds_valence, color='red')
        ax.plot(xs_depletion, ys_conduction, color='blue')
        ax.plot(xs_czts_conduction, ys_czts_conduction, color='blue')
        ax.plot(xs_cds_conduction, ys_cds_conduction, color='blue')
        ax.plot(xs_czts_fermi_p, ys_czts_fermi_p, color='green', linestyle='--')
        ax.plot(xs_cds_fermi_n, ys_cds_fermi_n, color='green', linestyle='--')
        ax.axvline(x=-x_n, color='gray', linestyle='--', label='Depletion Edge n')
        ax.axvline(x=x_p, color='gray', linestyle='--', label='Depletion Edge p')
        ax.axvline(x=-x_cds, color='black', label='n boundary')
        ax.axvline(x=x_czts, color='black', label='p boundary')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return
        
    def P_LayerFindRecombinationConstants(self):
        kt = const.k * const.T
        n = self.p_layer_n_0 + self.czts_delta_n
        p = self.p_layer_p_0 + self.czts_delta_p
        bar = self.p_layer.E_g / (520.57-143.76)
        #Energy levels
        E_v_cu = (150-143.76) * bar + self.p_layer.E_valence
        E_v_zn = (199-143.76) * bar + self.p_layer.E_valence
        E_v_sn1 = (229-143.76) * bar + self.p_layer.E_valence
        E_v_sn2 = (240-143.76)* bar + self.p_layer.E_valence
        E_v_sn3 = (260-143.76)* bar + self.p_layer.E_valence
        E_cu_zn = (174-143.76)* bar + self.p_layer.E_valence
        E_cu_sn1 = (218-143.76)* bar + self.p_layer.E_valence
        E_cu_sn2 = (250-143.76)* bar + self.p_layer.E_valence
        E_cu_sn3 = (283-143.76)* bar + self.p_layer.E_valence
        E_zn_sn = (204-143.76)* bar + self.p_layer.E_valence
        E_sn_cu1 = (331-143.76)* bar + self.p_layer.E_valence
        E_sn_cu2 = (393-143.76)* bar + self.p_layer.E_valence
        E_sn_cu3 = (519-143.76)* bar + self.p_layer.E_valence
        E_zn_cu = (483-143.76)* bar + self.p_layer.E_valence
        E_sn_zn1 = (480-143.76)* bar + self.p_layer.E_valence
        E_sn_zn2 = (472-143.76)* bar + self.p_layer.E_valence
        E_cu_i = (483-143.76)* bar + self.p_layer.E_valence
        E_zn_i = (472-143.76)* bar + self.p_layer.E_valence
        E_sn_i1 = (488-143.76)* bar + self.p_layer.E_valence
        E_sn_i2 = (346-143.76)* bar + self.p_layer.E_valence
        E_sn_i3 = (333-143.76)* bar + self.p_layer.E_valence
        E_v_s = (309-143.76)* bar + self.p_layer.E_valence
        xs = [0,1,2,2,2,3,4,4,4,5,6,6,6,7,8,8,9,10,11,11,11,12]
        self.state_level = [E_v_cu, E_v_zn, E_v_sn1,E_v_sn2,E_v_sn3, 
        E_cu_zn, E_cu_sn1,E_cu_sn2,E_cu_sn3, E_zn_sn, E_sn_cu1,E_sn_cu2,E_sn_cu3,
        E_zn_cu, E_sn_zn1,E_sn_zn2, E_cu_i, E_zn_i, E_sn_i1,E_sn_i2,E_sn_i3, E_v_s]
        #formation energies
        VCu = 0.77
        VZn = 1.12
        VSn1 = 2.82
        VSn2 = 2.82
        VSn3 = 2.82
        CuZn = 0.01
        CuSn1 = 0.87
        CuSn2 = 0.87
        CuSn3 = 0.87
        ZnSn = 0.69
        SnCu1 = 6.54
        SnCu2 = 6.54
        SnCu3 = 6.54
        ZnCu = 2.42
        SnZn1 = 4.11
        SnZn2 = 4.11
        Cui = 3.13 
        Zni = 5.92 
        Sni1 = 8.11
        Sni2 = 8.11
        Sni3 = 8.11
        Vs = 0.99
        formation_energies = [VCu, VZn, VSn1,VSn2,VSn3, CuZn, CuSn1,CuSn2,CuSn3, ZnSn, SnCu1,SnCu2,SnCu3, ZnCu, SnZn1,SnZn2, Cui, Zni, Sni1,Sni2,Sni3, Vs]
        construction_temp = 400 + 279.8 #kelvin
        construction_energy = const.k*construction_temp / const.q #eV
        
        N_t = np.logspace(22, 26, 10000)
        t_tot = np.zeros((len(N_t)))
        p_layer_t_tot = np.zeros((len(N_t)))
        p_layer_L_tot = np.zeros((len(N_t)))
        
        for k in range(len(N_t)):
            Z = 0
            for i in range(len(formation_energies)):
                Z += np.exp(-formation_energies[i]/ construction_energy )
            boltzmann_probabilities = np.zeros((len(formation_energies)))
            for i in range(len(formation_energies)):
                boltzmann_probabilities[i] = np.exp(- formation_energies[i] / construction_energy )/Z
            N_ts_boltzmann = boltzmann_probabilities*N_t[k]
            t_h_mins = np.zeros((len(xs)))
            t_e_mins = np.zeros((len(xs)))
            R = np.zeros((len(xs)))
            t = np.zeros((len(xs)))
            for i in range(len(xs)):
                t_h_mins[i] = 1 / (self.p_layer.sigma_h * self.p_layer.v_th * N_ts_boltzmann[i])
                t_e_mins[i] = 1 / (self.p_layer.sigma_e * self.p_layer.v_th * N_ts_boltzmann[i])
                R[i] =(n*p - self.p_layer.n_i**2) / (t_h_mins[i]*(n + self.p_layer.N_conduction*np.exp(- (self.p_layer.E_conduction-self.state_level[i])*const.q/kt)) + t_e_mins[i]*(p+self.p_layer.N_valence*np.exp(- (self.state_level[i]-self.p_layer.E_valence)*const.q/kt)))
                t[i] = self.czts_delta_n / R[i]
            t_tot[k] = 0
            for i in range(len(t)):
                t_tot[k] += 1/t[i]
            t_tot[k] = 1/t_tot[k]
        
            p_layer_srh_t_e = t_tot[k]
            p_layer_surf_t_e = self.pl_depth/(2*self.p_layer.v_surface)
            p_layer_rad_t_e = 1 / (self.p_layer.c_rad_n * self.p_layer_p_0)
            p_layer_aug_t_e = 1  /(self.p_layer.c_aug_n * self.p_layer_p_0**2)
            p_layer_t_tot[k] = 1 /(1/p_layer_srh_t_e + 1/p_layer_surf_t_e + 1/p_layer_rad_t_e + 1/p_layer_aug_t_e)
            p_layer_L_tot[k] = np.sqrt(p_layer_t_tot[k] * self.p_layer.diffusivity_n)
        #find value of N_t for which L_tot is correct
        L_actual = 1.4e-7
        #t_actual = 7.8e-9
        idx = (np.abs(p_layer_L_tot- L_actual)).argmin()  #index at v=0 for v ideal
        print('')
        print('p_layer_L_tot', p_layer_L_tot[idx])
        print('N_t', N_t[idx])
        print('')
    
    def N_LayerFindRecombinationConstants(self):
        kt = const.k * const.T
        n_i = self.n_layer.n_i
        n = self.n_layer_n_0 + self.cds_delta_n
        p = self.n_layer_p_0 + self.cds_delta_p
        E_t = self.n_layer.E_g*0.5 + self.n_layer.E_valence
        
        v_surface = np.logspace(-5, 4, 10000)
        n_layer_t_tot = np.zeros((len(v_surface)))
        n_layer_L_tot = np.zeros((len(v_surface)))
        
        for k in range(len(v_surface)):
            N_t = self.n_layer.N_T              #density of trapped states
            t_e_min = (1 / (self.n_layer.sigma_e * self.n_layer.v_th * N_t))
            t_h_min = (1 / (self.n_layer.sigma_h * self.n_layer.v_th * N_t))
            R_test = (n*p - n_i**2) / (t_h_min*(n + self.n_layer.N_conduction*np.exp(- (self.n_layer.E_conduction-E_t)*const.q/kt)) + t_e_min*(p+self.n_layer.N_valence*np.exp(- (E_t-self.n_layer.E_valence)*const.q/kt)))
            n_layer_srh_t_h = self.cds_delta_n / R_test
            n_layer_surf_t_h = self.nl_depth / (2 * v_surface[k])
            n_layer_rad_t_h = 1 / (self.n_layer.c_rad_p * self.n_layer_n_0)
            n_layer_aug_t_h = 1  /(self.n_layer.c_aug_p * self.n_layer_n_0**2)
            n_layer_t_tot[k] = 1 /(1/n_layer_srh_t_h + 1/n_layer_surf_t_h + 1/n_layer_rad_t_h + 1/n_layer_aug_t_h) 
            n_layer_L_tot[k] = np.sqrt(n_layer_t_tot[k] * self.n_layer.diffusivity_p)
        
        L_actual = 2.5e-7
        idx = (np.abs(n_layer_L_tot- L_actual)).argmin()  #index at v=0 for v ideal
        print('')
        print('n_layer_L_tot', n_layer_L_tot[idx])
        print('v_surface', v_surface[idx])
        print('')
