"""
JAX-FEM application for calculating parasitic capacitance.

This package provides tools to calculate parasitic capacitance between
conductors in VLSI/IC designs using the finite element method through JAX-FEM.
"""

try:
    from .capacitance import (
        ElectrostaticProblem,
        calculate_capacitance,
        read_gmsh_file
    )
except ImportError:
    # Fallback for when the package is not properly installed
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from applications.capacitance.capacitance import (
        ElectrostaticProblem,
        calculate_capacitance,
        read_gmsh_file
    )

__all__ = [
    'ElectrostaticProblem',
    'calculate_capacitance',
    'read_gmsh_file'
] 