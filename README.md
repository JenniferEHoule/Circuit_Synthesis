# Vector Fitting Algorithm and Circuit Sythesis
### Author: Jennifer Houle
### Date: 5/10/2020

This code was written for a class project in the course entitled ECE 504: "Modern Circuit Synthesis Algorithms" taught 
in Spring 2020 by Dr. Ata Zadehgol at the University of Idaho in Moscow. Dr. Zadehgol pointed out most of the relevant 
papers and resources listed under references, and provided overall project guidance as well as answered many questions.

# Overview
- This code implements the VFIT 3 (vector fitting) algorithm [1-3] for pole relocation based directly on the MATLAB code in [4]. 

- It also implements the passivity enforcement from [7-10]. This is directly based on the MATLAB code in [7]. The same 
input data restrictions apply.

- Finally, there is code for creating a netlist from poles and residues. This is based on the descriptions in [5-6].

- Flowcharts.pdf is a general flowchart of how this software can be used to model a system.

- A few more notes:
    * This has been tested for a limited range of input data. It produces similar results with the example data from [7].
    * If using a 'D' and 'E' for the fit described in [1-3], a shunt capacitor and/or resistor may be needed for the netlist.
    * The S Parameter method has not been implemented for passivity enforcement, unlike in [7]. Input parameters must be Y.
    * svd in Python appears much slower than in MATLAB. This program can take a long time depending on the machine it is run on
    for more than 4 ports. Any performance enhancements will likely start here.  
    * quadprog: this solver is more restrictive in Python than in MATLAB. There may be other versions that more closely align
    with MATLAB. When it works, which is most of the time, the results do appear consistent with MATLAB though.
    * If the S/Y-Parameter data is generated using ADS, the outputs can be saved in the MATLAB format and 
    `ads_mat_format_new.py` can be run to convert to the correct .npy format.

# Licensing
From [7], restrictions on use, in addition to licensing:

>- Embedding any of (or parts from) the routines of the Matrix Fitting Toolbox in a commercial software, or a software requiring licensing, is strictly prohibited. This applies to all routines, see Section 2.1.
>- If the code is used in a scientific work, then reference should me made as follows:
>  - VFdriver.m and/or vectfit3.m: References [1],[2],[3]
>  - RPdriver.m and/or FRPY.m applied to Y-parameters: [8],[9]

# Files:

## Main Program Files:
- convert_y_to_y_prime.py: Converts the Y matrix to a Y' matrix [6]
- vectfit3.py: Implements the vector fitting algorithm as a class, VectorFit3. [1-4]
    * Note: use asymp=AsympOptions.NONE for accurate netlist generation
- create_netlist.py: Use for generating a netlist from poles / residues [5, 6]
- VFdriver.py: Use for running the Vector Fitting Algorithm. Less manual option than vectfit3.py [7]
- RPdriver.py: Use to enforce passivity [7]
    * pass_check.py: Use to identify intervals that violate passivity [7], [10]
    * fitcalc.py:  Calculate Yfit from the state space model as in Eq. (5) in [10]. Based off [7]
    * FRPY.py: Perturb the model [7]
    * intercheig.py, rot.py: Adjust the eigenvalues / eigenvectors [7]
    * violextrema.py: Identify eigenvalue minima within given intervals [7]
    * plots.py: Contains several of the plots used
    * quadprog.py: This code is to replicate the quadprog function in MATLAB [11]
    * pr2ss.py: This function completes the state space model [7]
    * utils.py: Contains supporting functions including options found in [7]

## Supporting Program Files:
- Flowchart.pdf: Flowcharts, some results, etc.
- circuit_synthesis_env.yml: Version and library information
- ex2_Y.py: 3 port example from the Matrix Fitting Toolbox [7]
    * `ex2_Y_bigY.npy`, `ex2_Y_s.npy`
    the starting data for ex2_Y.py, converted from [7]
- 4port_example_ADS.py: 4 port example without passivity enforcement
    * `output_from_4port_simulation_original_s.npy`,`output_from_4port_simulation_original_f.npy` are
    the starting data generated from ADS for an LR circuit.
    * `output_from_4port_simulation_netlist_s.npy`,`output_from_4port_simulation_netlist_f.npy` are
    the data generated from ADS using the generated netlist.
- ads_mat_format_new.py: converts the ADS .mat output (saved as MATLAB file) to .np arrays
    * see `ADS.mat` for an example MATLAB output. 
    
# Full program
See example ex2_Y.py. This example:
1. Imports data. This is assumed to be Y' data (see flowchart in Flowchart file).
2. Runs vfdriver. This is the vector fitting algorithm. See Flowchart file for input details.
3. Runs rpdriver. This is the passivity enforcement. See Flowchard file for input details.
4. Runs create_netlist_file. This creates a netlist using the poles and residues. 
This will match the Y data, NOT the Y' data.

In order to obtain the Y' data, use the function convert_y_to_y_prime.py. The input will be a matrix with a row 
for each element Y11, Y12, ... YNN. Example usage for input data `f`:
```
bigY = convert_y_to_y_prime_matrix_values(f)
```

This work was done using ADS software for circuit simulations. Data can be saved directly to a MATLAB format, which
can then be read into Python. Initial data can be compared with data generated using the netlist created.

# Run Vector Fitting Algorithm
 To run, create an instance of the class `VectorFit3`, i.e.
`vector_fitter = VectorFit3(asymp='DE')`
    * A number of options are available, copied from [4], with the below being the defaults:
```
stable=True,			# Force unstable poles to be stable if True
# Options are None, D, DE selecting if d and e are used in Eq. (4) in [2]
asymp=AsympOptions.D,	
skip_pole=False,		# True will skip finding poles
skip_res=False,		# True will skip finding the residues
cmplx_ss=True,		# True will create a complex state space model
spy1=False,			# True will plot the sigma stage of vector fitting
spy2=True,			# True will plot the final fit data
logx=True,			# True will plot logarithmically on the x axis
logy=True,			# True will plot logarithmically on the y axis (does not apply to phase diagram)
errplot=True,			# True will display the difference between data and fit
phaseplot=False,		# True will plot the phase diagram
legend=True
```
To find the fit:
```
SER, poles, rmserr = vector_fitter.do_fit(f, s, poles, weight)
```

## Inputs:
- s: Frequency data in a vector. Frequency across which response data is given.
    * Dimension: number_of_frequency_points
- f: Frequency response data in a matrix. A row is created for each Y' parameter, with a response for each frequency. The number of rows is determined by the number of Y' parameters being evaluated. This may be a 1-D matrix.
    * Dimensions: number_of_Y'_responses X number_of_frequency_points
- poles: Initial guesses for the poles in a vector. These can be selected or autogenerated as in ex4a.py
    * Dimension: 1 X number_of_poles
- weight: Weighting matrix for the frequency response data. This may be set to 1's, or have a different weight per frequency. This may be 1-D so all Y' data is weighted in a common manner, or each Y' row may have its own weight.
    * Dimensions: number_of_Y'_responses X number_of_frequency_points
    * OR: 1 X number_of_frequency_points
    
Note the number_of_poles must be smaller than the number_of_frequency_points

## Outputs:
- SER: State space model in a dictionary:
    * A: Final poles in a diagonal matrix
        * Dimension: number_of_poles X number_of_poles
    * B: Vector of 1's
        * Dimension: number_of_poles X 1
    * C: Residues in a matrix (each row corresponds to a single Y' frequency response)
        * Dimension: number_of_Y'_responses X number_of_poles
    * D: d parameters for the fit (see Eq. (1) in [2])
        * Dimension: number_of_Y'_responses X 1
    * E: e parameters for the fit (see Eq. (1) in [2])
        * Dimension: number_of_Y'_responses X 1
- poles: new poles (Same information as `SER['A']` but in a vector)
    * Dimensions: 1 X number_of_poles
- rmserr: RMS error from the fit generated with SER    
    * Dimensions: number_of_iterations X 1
- Generated fit v. input f data graphs shown in magnitude / phase / difference depending on the options selected
  The graphs pop up and are saved as .png files.

# Usage:
The point of this program is to relocate poles. The do_fit method may be iterated, with the new poles calculated being used for the next iteration of the program. Successive iterations will allow the convergence of the poles.

Example code in which the plotting, including the phase, is only plotted on the final iteration. The residues are also only calculated on the final iteration:
```
Niter=5
rms = np.zeros((Niter, 1))
for iter in range(Niter):
    vector_fitter = VectorFit3(phaseplot=(iter == Niter - 1), skipres=(iter != Niter - 1), spy2=(iter == Niter - 1))
    SER, poles, rmserr = vector_fitter.do_fit(f, s, poles, weight)
    rms[iter,0] = rmserr`
```

# Python Version Information:
Python 3.7.6

Libraries used:
See circuit_synthesis_env.yml

# References:
```
[1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency
    domain responses by Vector Fitting", IEEE Trans. Power Delivery,
    vol. 14, no. 3, pp. 1052-1061, July 1999.

[2] B. Gustavsen, "Improving the pole relocating properties of vector
    fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
    July 2006.

[3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
    "Macromodeling of Multiport Systems Using a Fast Implementation of
    the Vector Fitting Method", IEEE Microwave and Wireless Components
    Letters, vol. 18, no. 6, pp. 383-385, June 2008.

[4] B. Gustavsen, VFIT3, The Vector Fitting Website. March 20, 2013. Accessed on:
    Jan. 21, 2020. [Online]. Available: 
    https://www.sintef.no/projectweb/vectfit/downloads/vfit3/.

[5] A. Zadehgol, "A semi-analytic and cellular approach to rational system characterization 
    through equivalent circuits", Wiley IJNM, 2015. [Online]. https://doi.org/10.1002/jnm.2119

[6] V. Avula and A. Zadehgol, "A Novel Method for Equivalent Circuit Synthesis from 
    Frequency Response of Multi-port Networks", EMC EUR, pp. 79-84, 2016. [Online]. 
    Available: ://WOS:000392194100012.

[7] B. Gustavsen, Matrix Fitting Toolbox, The Vector Fitting Website.
    March 20, 2013. Accessed on: Feb. 25, 2020. [Online]. Available:
    https://www.sintef.no/projectweb/vectorfitting/downloads/matrix-fitting-toolbox/.

[8] B. Gustavsen, "Fast passivity enforcement for S-parameter models by perturbation
    of residue matrix eigenvalues",
    IEEE Trans. Advanced Packaging, vol. 33, no. 1, pp. 257-265, Feb. 2010.

[9] B. Gustavsen, "Fast Passivity Enforcement for Pole-Residue Models by Perturbation
    of Residue Matrix Eigenvalues", IEEE Trans. Power Delivery, vol. 23, no. 4,
    pp. 2278-2285, Oct. 2008.

[10] A. Semlyen, B. Gustavsen, "A Half-Size Singularity Test Matrix for Fast and Reliable
    Passivity Assessment of Rational Models," IEEE Trans. Power Delivery, vol. 24, no. 1,
    pp. 345-351, Jan. 2009.

[11] divenex, Stack Overflow. Dec. 11, 2019. Accessed on: April 4, 2020.
    [Online]. Available: https://stackoverflow.com/a/59286910.

[12] stephane-caron, "Quadratic Programming in Python". Accessed on: May 3, 2020.
    [Online]. Available: https://scaron.info/blog/quadratic-programming-in-python.html.

[13] nolfwin, GitHub. March 11, 2018. Accessed on: May 3, 2020.
    [Online]. Available: https://github.com/nolfwin/cvxopt_quadprog/blob/master/cvxopt_qp.py.


```