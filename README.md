# Fokker-Planck: Python module for numerical solution and parameter inference for the 1D Fokker-Planck equation

## About

This python module contains tools for working with the **one-dimensional Fokker-Planck equation**

 <p align="center">
<img src="https://latex.codecogs.com/svg.image?\partial_t&space;P(x,t|x_0,t_0)&space;=&space;-&space;\partial_x&space;\left[&space;a(x,t)P(x,t|x_0,t_0)\right]&plus;&space;\partial_x^2&space;\left[&space;D(x,t)P(x,t|x_0,t_0)\right]&space;\qquad&space;(1)" title="\partial_t P(x,t|x_0,t_0) = - \partial_x \left[ a(x,t)P(x,t|x_0,t_0)\right]+ \partial_x^2 \left[ D(x,t)P(x,t|x_0,t_0)\right] \qquad (1),"/>
 </p>

where <i>P(x,t|x<sub>0</sub>,t<sub>0</sub>)</i> is the transition probability density to be at point <i>x</i> at time <i>t</i>, given that one started with delta-peak initial condition
 <i>P(x,t<sub>0</sub>|x<sub>0</sub>,t<sub>0</sub>) = &delta;(x-x<sub>0</sub>)</i> at point <i>x<sub>0</sub></i> at time <i>t<sub>0</sub></i>. Furthermore, <i>D(x,t)</i>, <i>a(x,t)</i> are the diffusivity and drift profile.

Currently implemented features are

* **Parameter inference** for time-independent diffusivity <i>D(x)</i> and drift <i>a(x)</i> from given realizations of the Langevin equation corresponding to Eq. (1) <a href="#ref_1">[1]</a>.
* **Numerical calculation of the spectrum** of the Fokker-Planck operator defined by the right-hand side of Eq. (1), for various boundary conditions (absorbing, no-flux/reflecting, periodic).
* **Numerical simulation** of Eq. (1) on a finite domain for various boundary conditions (absorbing, no-flux/reflecting, periodic, general robin boundary conditions) <a href="#ref_2">[2]</a>.


## Installation

To install the module fokker_planck, clone the repository and run the installation script:

```bash
>> git clone https://github.com/juliankappler/fokker-planck.git
>> cd fokker-planck
>> python setup.py install
```

## Examples

In the following we give some short examples of how the module is used. See the folder [examples/](examples/) for more detailed example notebooks.

### Parameter inference

Parameter inference is based on calculating the first two Kramers-Moyal coefficients, as explained in detail in Appendix A of <a href="#ref_1">Ref. [1]</a>. A detailed inference example is given [in this jupyter notebook](examples/inference/Parameter%20inference%20via%20Kramers-Moyal%20coefficients.ipynb), here we provide an abbreviated version of that example:

```Python

import fokker_planck

# filename of the pickle file that contains the trajectories
trajectories_filename = './sample_trajectories/sample_trajectories.pkl'
# the pickle file should contain a list of 1D arrays, i.e.
# trajectories = pickle.load(open(trajectories_filename,'rb'))
# must lead to an object "trajectories" such that
#     trajectories[i] = 1D array
# for i = 0, ..., len(trajectories)
# Note that a sample python script to generate data is provided in
#     examples/inference/sample_trajectories

# timestep of the trajectories
dt = 1e-4

# create a dictionary with the parameters
parameters = {'trajectories_filename':trajectories_filename,
            'dt':dt}

# create an instance of the kramers_moyal class
inference = fokker_planck.inference(parameters=parameters)

# load the trajectorial data (stored in a pickle file)
inference.load_trajectories()

# before running the inference, we need to create
# an index of the list of trajectories
inference.create_index()

# finally, we run the inference
inference_result = inference.run_inference()

# the call to run_inference returns a dictionary with three 1D arrays:
x = inference_result['x'] # positions
D = inference_result['D'] # diffusivity at the positions x
a = inference_result['a'] # drift at the positions x

```

Running this code on the sample data generated by [this script](examples/inference/sample_trajectories/generate_sample_trajectories.py), the inference yields results that compare very well to the diffusivity <i>D(x) = 1</i> and drift <i>a(x) = -8 &#183; (x<sup>2</sup>-1) &#183; x</i> used for sample data generation:

![Imagel](https://raw.githubusercontent.com/juliankappler/fokker-planck/master/examples/inference/inference-example.jpg)


### Numerical simulation

tbd

## References

<a id="ref_1">[1] **Experimental Measurement of Relative Path Probabilities and Stochastic Actions**. Jannes Gladrow, Ulrich F. Keyser, Ronojoy Adhikari, Julian Kappler. Physical Review X, vol. 11, p. 031022 (2021). DOI: [10.1103/PhysRevX.11.031022](https://doi.org/10.1103/PhysRevX.11.031022).</a>

<a id="ref_2">[2] **Stochastic action for tubes: Connecting path probabilities to measurement**. Julian Kappler, Ronojoy Adhikari. Physical Review Research, vol. 2, p. 023407, (2020). DOI: [10.1103/PhysRevResearch.2.023407](https://doi.org/10.1103/PhysRevResearch.2.023407).</a>
