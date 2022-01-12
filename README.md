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
* **Numerical simulation** of Eq. (1) on a finite domain for various boundary conditions (absorbing, no-flux/reflecting, periodic, general robin boundary conditions) <a href="#ref_2">[2]</a>.


## Installation

To install the module fractional_wave_equation, clone the repository and run the installation script:

```bash
>> git clone https://github.com/juliankappler/fokker-planck.git
>> cd fokker-planck
>> python setup.py install
```

## Examples

### Parameter inference

tbd

### Numerical simulation

tbd

## References

<a id="ref_1">[1] **Experimental Measurement of Relative Path Probabilities and Stochastic Actions**. Jannes Gladrow, Ulrich F. Keyser, Ronojoy Adhikari, Julian Kappler. Physical Review X, vol. 11, p. 031022 (2021). DOI: [10.1103/PhysRevX.11.031022](https://doi.org/10.1103/PhysRevX.11.031022).</a>

<a id="ref_2">[2] **Stochastic action for tubes: Connecting path probabilities to measurement**. Julian Kappler, Ronojoy Adhikari. Physical Review Research, vol. 2, p. 023407, (2020). DOI: [10.1103/PhysRevResearch.2.023407](https://doi.org/10.1103/PhysRevResearch.2.023407).</a>
