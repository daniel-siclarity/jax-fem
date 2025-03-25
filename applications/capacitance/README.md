# Parasitic Capacitance Calculation using JAX-FEM

This application calculates parasitic capacitance between conductors in VLSI/IC designs using the finite element method (FEM) through JAX-FEM. It solves the Poisson equation for electrostatics to find the electric potential distribution, then computes capacitance using the energy method.

## Features

- Reads GMSH (.msh) files containing IC geometries with conductors and dielectric regions
- Solves the Poisson equation for electrostatics with proper boundary conditions
- Calculates the capacitance matrix between multiple conductors
- Handles different dielectric materials with varying permittivity values
- Generates visualization files for electric potential distribution
- Robust handling of large meshes with many conductors

## Background

Parasitic capacitance in ICs refers to unintended capacitive coupling between conductors such as interconnects, transistor gates, and substrates. Accurately calculating these capacitances is crucial for:

1. Signal integrity analysis
2. Timing analysis
3. Power consumption estimation
4. Crosstalk prediction

The basic physics is governed by the Poisson equation:

```
∇·(ε∇φ) = -ρ
```

Where:
- φ is the electric potential
- ε is the permittivity (can vary by material)
- ρ is the charge density (often zero in dielectrics)

Once the potential distribution is obtained, the capacitance matrix can be calculated using the energy method:

```
C_ij = 2*E_ij / V_i*V_j
```

Where:
- E_ij is the electrostatic energy when conductor i is at potential V_i and conductor j is at potential V_j
- The energy is calculated as E = (1/2)∫εE²dV

## Usage

### Basic Usage

To calculate parasitic capacitance for a GMSH file:

```bash
python -m applications.capacitance.capacitance path/to/your/file.msh --output output_directory
```

### Demo

The application includes a demo script that generates simple capacitor structures and calculates their capacitance:

```bash
# Generate a parallel plate capacitor and calculate capacitance
python -m applications.capacitance.demo --type parallel --output demo_output

# Generate an interdigitated capacitor and calculate capacitance
python -m applications.capacitance.demo --type interdigitated --output demo_output
```

### Using with IC Layouts

1. Export your IC layout geometry to GMSH format (.msh)
2. Run the capacitance calculation application
3. View the results in the output directory

## Input File Format

The application expects a GMSH file (.msh) with:

1. A mesh representation of the IC geometry
2. Physical groups identifying conductors and dielectric regions
3. Proper boundary conditions

For large designs with many conductors, the tool can be configured to only analyze a subset of conductors to reduce computation time.

## How It Works

The implementation follows these key steps:

1. **Mesh Reading**: Parses GMSH files to extract geometry, conductors, and material properties
2. **Material Assignment**: Maps dielectric constants to mesh elements based on material regions
3. **Boundary Setup**: Sets up Dirichlet boundary conditions (fixed potentials) on conductor surfaces
4. **Direct Solution**: For efficiency, we directly apply boundary conditions without using a full solver
5. **Surface Charge Calculation**: Computes surface charge on conductors from the electric field gradient
6. **Capacitance Matrix**: Constructs the capacitance matrix from the potential/charge relationship

The improved implementation handles edge cases robustly:
- Empty or missing material properties
- Very large meshes with many conductors
- Complex geometry with multiple dielectric regions

## Output

The application produces:

1. A text file containing the capacitance matrix
2. Log information about the simulation

## Requirements

- JAX-FEM
- NumPy
- GMSH (for generating meshes)
- Meshio (for reading/writing mesh files)

## Recent Improvements

The code has been enhanced to:
- Fix issues with material initialization
- Handle JAX tracing errors with boundary conditions
- Improve robustness with proper error handling
- Add optimizations for large meshes with many conductors
- Fix handling of surface elements for charge calculations

## Limitations and Future Work

Current limitations:
- The implementation uses a simplified electric field calculation
- Material interfaces may need refinement for accurate solutions
- Large 3D meshes can be computationally expensive

Future improvements:
- Better handling of material interfaces
- Full solver-based approach for more accurate results
- GPU acceleration for large meshes 
- Integration with SPICE for circuit simulation
- Direct import from GDS-II or other IC layout formats

## References

1. Xue, Tianju, et al. "JAX-FEM: A differentiable GPU-accelerated 3D finite element solver for automatic inverse design and mechanistic data science." Computer Physics Communications (2023): 108802.
2. N. P. van der Meijs and A. J. van Genderen, "An efficient finite element method for submicron IC capacitance extraction," 33rd Design Automation Conference, 1996. 