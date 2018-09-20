import numpy as np
import Constants as const

class Material:
    '''Define a new material to use as a layer in the solar cell'''
    def __init__(self, name, data_file=None):
        if name=='air':
            self.name = name
            self.refractive_index = const.n_air
        else:
            self.name = name
            #import data on material
            self.data_file = data_file
            self.header_skip = self.get_header_skip(self.name)
            if self.name == 'mo' or self.name=='tio2' or self.name=='sio2':
                self.data = self.get_data2()
            if self.name == 'czts' or self.name=='cds' or self.name=='zno':
                self.data = self.get_data1(self.data_file, self.header_skip)
            self.energy = self.data[0] # eV
            self.wavelength = self.data[1] # nm
            self.e1 = self.data[2]
            self.e2 = self.data[3]
            self.refractive_index = self.data[4]
            self.k = self.data[5]
            self.absorption_coeff = self.data[6] # m^-1
            self.reflection_coeff = self.data[7]
            self.resistivity = self.get_resistivity() #Ohm meters
            
            #p/n materials
            #values of 1 are currently unknown, values marked by '#' are estimates
            if self.name == 'czts':
                #recombination
                self.c_rad_p = 2*1e-10 *1e-6           #radiative recombination constant (m^3/s)
                self.c_rad_n = 2*1e-10 * 1e-6          
                self.c_aug_p = 7e-30 *1e-12            #Auger recombination constant (m^6 / s)
                self.c_aug_n = 7e-30 *1e-12
                self.v_surface = 5500e-2               #Surface velocity (m / s) 
                self.v_th = 1e5                        #thermal velocity of electrons (m/s)
                self.sigma_e = 1e-19                   #electron capture cross-section  (m^2)
                self.sigma_h = 1e-19                   #hole capture cross-section (m^2)
                self.sigma_surf_e = 1
                self.sigma_surf_h = 1
                self.n_surf_trap = 1
                #Material Properties
                self.N_T = 4.276e+24                    #density of trapped states (m^-3)
                self.mobility_h = 25e-4                 #hole mobility (m^2 s^-1 V-1)
                self.mobility_e = 7.25e-4 #6.2e-4       #electron mobility (m^2 s^-1 V-1) 
                self.diffusivity_n = self.mobility_e * const.k*const.T/const.q  #(m^2 / s)
                self.diffusivity_p = self.mobility_h * const.k*const.T/const.q  #(m^2 / s)
                self.e_r = 7.5                          #relative permitivity of czts
                self.e = self.e_r*const.e 
                self.N_conduction = 2.2e18*1e6          #DOS in conduction band (m^-3) 
                self.N_valence = 1.8e19*1e6             #DOS in valence band (m^-3)
                #band edges and energies (undoped)
                self.E_g = 1.32                         #Energy gap (eV)
                self.E_valence = 0
                self.E_conduction = self.E_valence + self.E_g
                self.E_fermi = self.E_valence + 0.5*self.E_g - 0.5*const.k*const.T/const.q*np.log(self.N_conduction/self.N_valence)   #undoped fermi energy in equilibrium (eV)
                self.n_i = np.sqrt(self.N_conduction*self.N_valence*np.exp(-self.E_g*const.q/(const.k*const.T))) #intrinsic density (when not doped)
                self.setup_data = {'c_rad_p':self.c_rad_p, 
                                   'c_rad_n':self.c_rad_n,
                                   'c_aug_p':self.c_aug_p, 
                                   'c_aug_n':self.c_aug_n,
                                   'v_th':self.v_th,
                                   'sigma_e':self.sigma_e, 
                                   'sigma_h':self.sigma_h,
                                   'sigma_surf_e':self.sigma_surf_e, 
                                   'sigma_surf_h':self.sigma_surf_h,
                                   'N_T_surf':self.n_surf_trap,
                                   'v_surf':self.v_surface,
                                   'N_T':self.N_T,
                                   'mobility_h':self.mobility_h,
                                   'mobility_e':self.mobility_e,
                                   'diffusivity_n':self.diffusivity_n,
                                   'diffusivity_p':self.diffusivity_p,
                                   'e_r':self.e_r,
                                   'e':self.e,
                                   'N_conduction':self.N_conduction,
                                   'N_valence':self.N_valence,
                                   'E_g':self.E_g,
                                   'E_valence':self.E_valence,
                                   'E_conduction':self.E_conduction,
                                   'E_fermi':self.E_fermi,
                                   'n_i':self.n_i}
            if self.name == 'cds':
                #recombination
                self.c_rad_p = 2*1e-10 *1e-6           #radiative recombination constant (m^3/s)
                self.c_rad_n = 2*1e-10 * 1e-6          
                self.c_aug_p = 7e-30 *1e-12            #Auger recombination constant (m^6 / s)
                self.c_aug_n = 7e-30 *1e-12
                self.v_th = 1e5                        #thermal velocity of electrons m/s
                self.sigma_e = 1e-19                   #electron capture cross-section  m^2
                self.sigma_h = 1e-19                   #hole capture cross-section     m^2
                self.sigma_surf_e = 1
                self.sigma_surf_h = 1e-20 * 1e4 #
                self.n_surf_trap = 6e14 * 1e-4
                self.v_surface = 0.727
                #Material Properties
                self.N_T = 1e23                        #density of trapped states (m^-3)
                self.mobility_h = 25e-4                #hole mobility (m^2 s^-1 V-1)
                self.mobility_e = 100e-4               #electron mobility (m^2 s^-1 V-1)
                self.diffusivity_n = self.mobility_e * const.k*const.T/const.q  #(m^2 / s)
                self.diffusivity_p = self.mobility_h * const.k*const.T/const.q  #(m^2 / s)
                self.e_r = 5.16                        #relative permitivity of cds
                self.e = self.e_r * const.e                
                self.N_conduction = 1.8e19*1e6 #       #DOS in conduction band (m^-3)
                self.N_valence = 2.4e18*1e6 #          #DOS in valence band (m^-3)
                #band edges and energies (undoped)
                self.E_g = 2.5                         #Energy gap (eV)
                self.E_valence = 0
                self.E_conduction = self.E_valence + self.E_g
                self.E_fermi = self.E_valence + 0.5*self.E_g - 0.5*const.k*const.T/const.q*np.log(self.N_conduction/self.N_valence)     #undoped fermi energy in equilibrium (eV)
                self.n_i = np.sqrt(self.N_conduction*self.N_valence*np.exp(-self.E_g*const.q/(const.k*const.T))) #intrinsic density (when not doped)
                self.setup_data = {'c_rad_p':self.c_rad_p, 
                                   'c_rad_n':self.c_rad_n,
                                   'c_aug_p':self.c_aug_p, 
                                   'c_aug_n':self.c_aug_n,
                                   'v_th':self.v_th,
                                   'sigma_e':self.sigma_e, 
                                   'sigma_h':self.sigma_h,
                                   'sigma_surf_e':self.sigma_surf_e, 
                                   'sigma_surf_h':self.sigma_surf_h,
                                   'N_T_surf':self.n_surf_trap,
                                   'v_surf':self.v_surface,
                                   'N_T':self.N_T,
                                   'mobility_h':self.mobility_h,
                                   'mobility_e':self.mobility_e,
                                   'diffusivity_n':self.diffusivity_n,
                                   'diffusivity_p':self.diffusivity_p,
                                   'e_r':self.e_r,
                                   'e':self.e,
                                   'N_conduction':self.N_conduction,
                                   'N_valence':self.N_valence,
                                   'E_g':self.E_g,
                                   'E_valence':self.E_valence,
                                   'E_conduction':self.E_conduction,
                                   'E_fermi':self.E_fermi,
                                   'n_i':self.n_i}
    
    def change_energy_levels(self, shift):
        '''Increase all energy levels by shift (eV)'''
        self.E_valence += shift
        self.E_conduction += shift
        self.E_fermi += shift

    def get_resistivity(self):
        '''Get material resistivity (ohm m)'''
        if self.name == 'mo':
            resistivity = 11.1 *1e-5 #https://www.sciencedirect.com/science/article/pii/S0040609014007731
        if self.name=='tio2':
            resistivity = 5 * 1e-2   #https://www.hindawi.com/journals/acmp/2013/365475/
        if self.name=='sio2':
            resistivity = 1
        if self.name == 'cds':
            resistivity = 10e-3     #https://www.sciencedirect.com/science/article/pii/0379678787900676
        if self.name == 'czts':
            resistivity = 2.53e-5   #https://www.sciencedirect.com/science/article/pii/S2187076415300245
        if self.name == 'zno':
            resistivity = 1.4 *1e-6 #https://www.sciencedirect.com/science/article/pii/S0040609005014550
        return resistivity
    
    def change_data(self, new_data):
        '''Change all the data of a material to new_data'''
        self.data = new_data
        self.energy = new_data[0]
        self.wavelength = new_data[1]
        self.e1 = new_data[2]
        self.e2 = new_data[3]
        self.refractive_index = new_data[4]
        self.k = new_data[5]
        self.absorption_coeff = new_data[6]
        self.reflection_coeff = new_data[7]
    
    def get_data1(self, data_file, header_skip):
        '''Imports data on material. crops data to max energy of solar spectrum'''
        data_long = np.genfromtxt(data_file, unpack=True, delimiter=' ', skip_header=header_skip)
        cut_off = 0
        E_max = 4.4336912385446254
        data_long[6] = data_long[6] * 100  # absorption coeff from cm^1 to m^-1
        
        for i in range(len(data_long[0])):
            if data_long[0][i] > E_max:
                cut_off = i
                break 
        delete = np.arange(cut_off, len(data_long[0]))
        data = np.zeros((len(data_long),cut_off))
        for i in range(len(data_long)):
            data[i] = np.delete(data_long[i], delete)
        return data
        
    def get_data2(self):
        '''Imports data on mo, tio2, sio2. crops data to max energy of solar spectrum'''
        data_rev = np.genfromtxt(self.data_file, unpack=True, delimiter=' ', skip_header=self.header_skip)
        cut_off = 0
        E_max = 4.4336912385446254
        
        wavelength = data_rev[0][::-1] *1e3   #nm, long to short wavelength
        e1 = np.zeros((len(wavelength)))      #no e1 data
        e2 = np.zeros((len(wavelength)))      #no e2 data
        refractive_index = data_rev[1][::-1]
        k = data_rev[2][::-1]
        wavelength_meters = wavelength * 1e-9
        absorption_coeff = 4 * np.pi * k / wavelength_meters  # m^-1
        reflection_coeff = np.zeros((len(wavelength)))
        energy = const.h * const.c / (wavelength*1e-9 * const.q) # eV

        data_long = np.array((energy, wavelength, e1, e2, refractive_index, k, absorption_coeff, reflection_coeff))
        for i in range(len(data_long[0])):
            if data_long[0][i] > E_max:
                cut_off = i
                break 
        delete = np.arange(cut_off, len(data_long[0]))
        data = np.zeros((8,cut_off))
        for i in range(len(data_long)):
            data[i] = np.delete(data_long[i], delete)
        return data
    
    def get_header_skip(self, name):
        '''Get the number of header lines to skip during data import'''
        if self.name == 'cds':
            header_skip = 2
        elif self.name == 'czts':
            header_skip = 2
        elif self.name == 'mo':
            header_skip = 3
        else:
            header_skip = 2
        return header_skip

    def get_energy_gap_with_T(self, T_L):
        '''Get energy gap using varshni's parameters (eV)'''
        if self.name == 'czts':
            E_g_0 = 1.64                      #(eV) E_g at 0K
            varsh_alpha = 1e-3                #(eV / K)
            varsh_beta = 340                  #(K)  closely related to Debye temp
        else:
            E_g_0 = 1.64                      #(eV) E_g at 0K
            varsh_alpha = 1e-3                #(eV / K)
            varsh_beta = 340                  #(K)  closely related to Debye temp
        E_g = E_g_0 - (varsh_alpha * T_L**2)/(varsh_beta + T_L)
        return E_g

 
