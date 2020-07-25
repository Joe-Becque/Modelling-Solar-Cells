# Modelling Solar Cells

This program was created for the purpose of modelling optimised thin film solar cells.
The publication of this work can be found at [https://doi.org/10.1088/1361-6404/aaf954](https://doi.org/10.1088/1361-6404/aaf954)

## Summary

This program models TiO2/ZnO/CdS/Cu2ZnSnS4/Mo solar cells with a p doped Cu2ZnSnS4 absorber
layer and an n doped CdS buffer layer under AM1.5 global solar illumination.

It tests solar cells with different absorber layer and buffer layer depths and doping
concentrations to measure the impact this has on the efficiency.

The simulation accounts for optical losses within the device. Additionally, losses due to
Shockley Read Hall recombination, Auger recombination, radiative recombination, and surface 
recombination are accounted for using a shunt resistance.

Provided with the data defining more materials' optical properties, this model has the potential
to be extended to simulate any thin film solar cell.


## In this Repository:

CdS_data,<br/>
CZTS_data,<br/>
Mo_data2,<br/>
TiO2_data,<br/>
and Zn0_data:<br/>
* Contain data on each solar cell layer's optical properties.

PVL_Spectrum.csv:
* Contains numerical data for a solar AM1.5 spectrum.

Spectrum.py
* Initialises a solar spectrum with which to illuminate the solar cell.

Material.py
* Initialises a layer in the material.

Solarcell.py
* Creates and models a solar cell with layers made from Materials specified and under the illumination of a spectrum.
* Adjusts the input data describing the Materials and the Spectrum so that they are compatible witheach other.
* Finds the energy levels in the p-n junction when the solar cell is formed.
* Calculates the external quantum efficiency of the solar cell by modelling the light transmitted through each layer of the solar cell, and the light absorbed in the CdS and CZTS layer of the solar cell.
* Finds the generation rate of carriers in the p-n junction as a function of depth through the layers.
* Calculates the shunt resistance by modelling recombination mechanisms in the solar cell.
* Finds the voltage-current relation at a load resistance by considering an equivalent circuit.
* Finds the maximum possible power output and efficiency of the solar cell by choosing an optimum load resistance.

GenerateCells.py:
* Generates solar cells by varying the doping concentrations and depths over a range and frequency specified.
* Depths are varied independently at each set of doping concentrations specified.
* Doping concentrations are varied independently at each set of depths specified.
* Returns Dopingfile.txt and Depthsfile.txt which contain all the data on every solar cell the program tests.

Run.py:
* Runs 2 instances of GenerateCells.py to collect data used for the project.

Constants.py:
* Physical constants.


## Run

Compiling Run.py will run the program and collect data on simulated solar cells.
Change parameters in Run.py to specify the solar cell layers and the number of solar cells simulated.
Note that even over a small set of depths and doping concentrations the runtime can become long.

