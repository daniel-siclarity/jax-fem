# JAX-FEM Parasitic Capacitance

This repository is a fork of [JAX-FEM](https://github.com/deepmodeling/jax-fem) with a focus on parasitic capacitance estimation for integrated circuits and VLSI designs.

## Parasitic Capacitance Estimation

The main application in this fork is a parasitic capacitance estimation tool that:

- Calculates capacitance matrices between conductors in IC designs
- Supports complex 3D geometries from GMSH files
- Handles multiple dielectric materials
- Scales to handle large meshes with many conductors

For details on using the capacitance estimation tool, see [applications/capacitance/README.md](applications/capacitance/README.md).

## Quick Start

To use the parasitic capacitance estimation tool:

```bash
# Run with a simple parallel plate capacitor demo
python -m applications.capacitance.demo --type parallel --output demo_output

# Run with a real IC design from a GMSH file
python -m applications.capacitance.capacitance path/to/your/design.msh --output capacitance_results
```

## About JAX-FEM

JAX-FEM is a GPU-accelerated differentiable finite element analysis package based on [JAX](https://github.com/google/jax). Please see the original repository at [deepmodeling/jax-fem](https://github.com/deepmodeling/jax-fem) for more details about the core library.

## Installation

Create a conda environment from the given [`environment.yml`](https://github.com/deepmodeling/jax-fem/blob/main/environment.yml) file and activate it:

```bash
conda env create -f environment.yml
conda activate jax-fem-env
```

Install JAX:
- See jax installation [instructions](https://github.com/jax-ml/jax?tab=readme-ov-file#installation). Depending on your hardware, you may install the CPU or GPU version of JAX.

Clone this repository and install locally:

```bash
git clone https://github.com/your-username/jax-fem-parasitic.git
cd jax-fem-parasitic
pip install -e .
```

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE](https://www.gnu.org/licenses/) for details.

## Citations

If you found this library useful in academic or industry work, please consider citing the original JAX-FEM paper:

```bibtex
@article{xue2023jax,
  title={JAX-FEM: A differentiable GPU-accelerated 3D finite element solver for automatic inverse design and mechanistic data science},
  author={Xue, Tianju and Liao, Shuheng and Gan, Zhengtao and Park, Chanwook and Xie, Xiaoyu and Liu, Wing Kam and Cao, Jian},
  journal={Computer Physics Communications},
  pages={108802},
  year={2023},
  publisher={Elsevier}
}
```

## Finite Element Method (FEM) [Original README.md content from here now for more detailed reference and jumpstart]
![Github Star](https://img.shields.io/github/stars/deepmodeling/jax-fem)
![Github Fork](https://img.shields.io/github/forks/deepmodeling/jax-fem)
![License](https://img.shields.io/github/license/deepmodeling/jax-fem)

FEM is a powerful tool, where we support the following features

- 2D quadrilateral/triangle elements
- 3D hexahedron/tetrahedron elements
- First and second order elements
- Dirichlet/Neumann/Robin boundary conditions
- Linear and nonlinear analysis including
  - Heat equation
  - Linear elasticity
  - Hyperelasticity
  - Plasticity (macro and crystal plasticity)
- Differentiable simulation for solving inverse/design problems __without__ deriving sensitivities by hand, e.g.,
  - Topology optimization
  - Optimal thermal control
- Integration with PETSc for solver choices

**Updates** (Dec 11, 2023):

- We now support multi-physics problems in the sense that multiple variables can be solved monolithically. For example, consider running  `python -m applications.stokes.example`
- Weak form is now defined through  volume integral and surface integral. We can now treat body force, "mass kernel" and "Laplace kernel" in a unified way through volume integral, and treat "Neumann B.C." and "Robin B.C." in a unified way through surface integral. 

<p align="middle">
  <img src="images/ded.gif" width="600" />
</p>
<p align="middle">
    <em >Thermal profile in direct energy deposition.</em>
</p>

<p align="middle">
  <img src="images/von_mises.png" width="400" />
</p>
<p align="middle">
    <em >Linear static analysis of a bracket.</em>
</p>

<p align="middle">
  <img src="images/polycrystal_grain.gif" width="360" />
  <img src="images/polycrystal_stress.gif" width="360" />
</p>
<p align="middle">
    <em >Crystal plasticity: grain structure (left) and stress-xx (right).</em>
</p>

<p align="middle">
  <img src="images/stokes_u.png" width="360" />
  <img src="images/stokes_p.png" width="360" />
</p>
<p align="middle">
    <em >Stokes flow: velocity (left) and pressure(right).</em>
</p>

<p align="middle">
  <img src="images/to.gif" width="600" />
</p>
<p align="middle">
    <em >Topology optimization with differentiable simulation.</em>
</p>

## Tutorial

| Example                                                      | Highlight                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [poisson](https://github.com/deepmodeling/jax-fem/tree/main/demos/poisson) | $${\color{green}Basics:}$$  Poisson's equation in a unit square domain with Dirichlet and Neumann boundary conditions, as well as a source term. |
| [linear_elasticity](https://github.com/deepmodeling/jax-fem/tree/main/demos/linear_elasticity) | $${\color{green}Basics:}$$  Bending of a linear elastic beam due to Dirichlet and Neumann boundary conditions. Second order tetrahedral element (TET10) is used. |
| [hyperelasticity](https://github.com/deepmodeling/jax-fem/tree/main/demos/hyperelasticity) | $${\color{blue}Nonlinear \space Constitutive \space Law:}$$ Deformation of a hyperelastic cube due to Dirichlet boundary conditions. |
| [plasticity](https://github.com/deepmodeling/jax-fem/tree/main/demos/plasticity) | $${\color{blue}Nonlinear \space Constitutive \space Law:}$$ Perfect J2-plasticity model is implemented for small deformation theory. |
| [phase_field_fracture](https://github.com/deepmodeling/jax-fem/tree/main/demos/phase_field_fracture) | $${\color{orange}Multi-physics \space Coupling:}$$ Phase field fracture model is implemented. Staggered scheme is used for two-way coupling of displacement field and damage field. Miehe's model of spectral decomposition is implemented for a 3D case. |
| [thermal_mechanical](https://github.com/deepmodeling/jax-fem/tree/main/demos/thermal_mechanical) | $${\color{orange}Multi-physics \space Coupling:}$$ Thermal-mechanical modeling of metal additive manufacturing process. One-way coupling is implemented (temperature affects displacement). |
| [thermal_mechanical_full](https://github.com/deepmodeling/jax-fem/tree/main/demos/thermal_mechanical_full) | $${\color{orange}Multi-physics \space Coupling:}$$ Thermal-mechanical modeling of 2D plate. Two-way coupling (temperature and displacement) is implemented with a monolithic scheme. |
| [wave](https://github.com/deepmodeling/jax-fem/tree/main/demos/wave) | $${\color{lightblue}Time \space Dependent \space Problem:}$$ The scalar wave equation is solved with backward difference scheme. |
| [topology_optimization](https://github.com/deepmodeling/jax-fem/tree/main/demos/topology_optimization) | $${\color{red}Inverse \space Problem:}$$ SIMP topology optimization for a 2D beam. Note that sensitivity analysis is done by the program, rather than manual derivation. |
| [inverse](https://github.com/deepmodeling/jax-fem/tree/main/demos/inverse) | $${\color{red}Inverse \space Problem:}$$ Sanity check of how automatic differentiation works. |
| [plasticity_gradient](https://github.com/deepmodeling/jax-fem/tree/main/applications/plasticity_gradient) | $${\color{red}Inverse \space Problem:}$$ Automatic sensitivity analysis involving history variables such as plasticity. |
