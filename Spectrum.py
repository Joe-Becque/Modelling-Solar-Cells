import numpy as np
import Constants as const

class Spectrum:
    def __init__(self, name):
        self.name = name
        spectrum_data = self.get_spectrum_data()
        self.energy = spectrum_data[0]           #eV
        self.irradiance = spectrum_data[1]       #W * m^-2 * nm^-1
        self.wavelength = spectrum_data[2]       #nm
        self.energy_joules = spectrum_data[3]    #J
        self.photocurrent = spectrum_data[4]     #photons * m^-2 * nm^-1 * s^-1
    
    def get_spectrum_data(self):
        '''Import AM1.5 Solar Spectrum.
           Irradiance and photocurrent as a function of photon energy and wavelength.'''
        if self.name == 'pvl':
            all_data = np.genfromtxt('PVL_Spectrum.csv', unpack=True, delimiter=',', skip_header=1)
            #https://www2.pvlighthouse.com.au/resources/optics/spectrum%20library/spectrum%20library.aspx
            wavelength_rev = all_data[0]                    #nm
            wavelength = wavelength_rev[::-1]
            irradiance_rev = all_data[1]                    #W * m^(-2) * nm^(-1)
            irradiance = irradiance_rev[::-1]
        elif self.name == 'etr':
            all_data = np.genfromtxt('ASTMG173.csv', unpack=True, delimiter=',', skip_header=2)
            wavelength = all_data[0]                    #nm
            irradiance = all_data[1]                    #W * m^(-2) * nm^(-1)
        elif self.name == 'gt':
            all_data = np.genfromtxt('ASTMG173.csv', unpack=True, delimiter=',', skip_header=2)
            wavelength = all_data[0]                    #m
            irradiance = all_data[2]                    #W * m^(-2) * nm^(-1)
        elif self.name == 'dc':
            all_data = np.genfromtxt('ASTMG173.csv', unpack=True, delimiter=',', skip_header=2)
            wavelength = all_data[0]                    #m
            irradiance = all_data[2]                    #W * m^(-2) * nm^(-1)
        
        E_eV = const.h * const.c / (wavelength * 10**(-9) * const.q)  #energy of photon at each wavelength (eV)
        E_joules = const.h * const.c / (wavelength * 10**(-9))        #energy of photon at each wavelength (J)
        photocurrent = irradiance / E_joules
        return E_eV, irradiance, wavelength, E_joules, photocurrent
        
