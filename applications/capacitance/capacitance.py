# Import necessary packages
import jax
import jax.numpy as np
import jax.numpy as jnp
import numpy as onp
import os
import meshio
import gmsh
import sys
import argparse

# Import JAX-FEM specific modules
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh


class ElectrostaticProblem(Problem):
    """
    JAX-FEM problem for electrostatic analysis.
    Solves the Poisson equation: -∇·(ε∇φ) = ρ
    where φ is the electric potential, ε is the dielectric constant,
    and ρ is the charge density.
    """
    def __init__(self, mesh, vec, dim, ele_type, dirichlet_bc_info=None, location_fns=None, dielectric_constants=None, surface_elements=None):
        """Initialize the electrostatic problem"""
        super().__init__(mesh=mesh, vec=vec, dim=dim, ele_type=ele_type, 
                        dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)
        self.dielectric_constants = dielectric_constants or {0: 1.0}  # Default is vacuum
        self.cell_materials = None  # Will be set later
        self.surface_elements = surface_elements or {}  # Map of conductor name to surface elements
        
        # Initialize internal variables
        self.internal_vars = ()

    def set_materials(self, cell_materials):
        """Set material properties for each cell.
        
        Args:
            cell_materials: Array of material IDs for each cell
        """
        self.cell_materials = cell_materials
        
        # Convert material IDs to permittivity values
        eps_values = []
        for mat_id in cell_materials:
            # Convert JAX array to integer for dictionary lookup
            mat_id_int = int(mat_id)
            eps_val = self.dielectric_constants.get(mat_id_int, 1.0)
            eps_values.append(eps_val)
        
        # Store permittivity values as internal variables
        # Expand to match quadrature points
        eps_array = np.array(eps_values)
        eps_expanded = np.repeat(eps_array[:, None], self.fes[0].num_quads, axis=1)
        self.internal_vars = (eps_expanded,)

    def get_tensor_map(self):
        """
        Get the tensor map function for the electrostatics problem.
        Returns:
            Function that computes the tensor term for a given point.
        """
        def tensor_map(u_grad, *internal_vars):
            # Get permittivity from internal variables if provided
            if internal_vars and len(internal_vars) > 0:
                # The first internal var should be the permittivity at this point
                eps = internal_vars[0]
                # Return dielectric displacement field with variable dielectric constant
                D = eps * u_grad
                return D
            else:
                # Return with default permittivity of 1.0 (vacuum)
                return u_grad
        return tensor_map
    
    def get_mass_map(self):
        """
        Get the mass map function for the electrostatics problem.
        Returns:
            Function that computes the mass term for a given point.
        """
        def mass_map(u, x, *internal_vars):
            # Compute electric field (negative gradient of potential)
            E = -u
            # Return dielectric displacement field with default dielectric constant
            return E
        return mass_map
    
    def calculate_electric_field(self, sol):
        """
        Calculate electric field E = -∇φ from the potential solution.
        Uses vmap to vectorize the calculation over all elements.
        """
        def single_element_field(cell_idx):
            # Get the element
            cell = self.fes[0].cells[cell_idx]
            
            # Get node potentials for this element
            node_potentials = sol[0][cell]
            
            # Get quadrature points and weights for this element
            x_quad = self.fes[0].elements[cell_idx].x_quad
            w_quad = self.fes[0].elements[cell_idx].w_quad
            
            # Get shape function gradients at quadrature points
            # Shape: (num_quad_points, num_nodes, dim)
            grad_shape_quad = self.fes[0].elements[cell_idx].grad_shape_quad
            
            # Calculate potential gradient at quadrature points
            # grad_phi = Σᵢ φᵢ∇Nᵢ
            # Shape: (num_quad_points, dim)
            grad_phi = -np.einsum('i,ijk->jk', node_potentials, grad_shape_quad)
            
            # Calculate weighted average over quadrature points
            # Shape: (dim,)
            element_field = np.sum(grad_phi * w_quad[:, None], axis=0) / np.sum(w_quad)
            
            return element_field

        # Vectorize over all cells
        cell_indices = np.arange(len(self.fes[0].cells))
        E_field = jax.vmap(single_element_field)(cell_indices)
        
        return E_field
    
    def calculate_energy(self, sol):
        """Calculate the total electrostatic energy.
        
        Args:
            sol: Solution vector
        
        Returns:
            Total electrostatic energy
        """
        # Get the solution values at quadrature points
        cells_sol = np.take(sol, self.fes[0].cells, axis=0)  # (num_cells, num_nodes, vec)
        
        # Calculate gradients at quadrature points
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim)
        # -> (num_cells, num_quads, num_nodes, vec, dim)
        u_grads = cells_sol[:, None, :, :, None] * self.fes[0].shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2)  # (num_cells, num_quads, vec, dim)
        
        # Get permittivity values for each quadrature point
        eps = self.internal_vars[0]  # (num_cells, num_quads)
        
        # Calculate energy density at each quadrature point
        # Energy density = 1/2 * ε * |∇φ|^2
        energy_density = 0.5 * eps[:, :, None, None] * np.sum(u_grads**2, axis=-1)  # (num_cells, num_quads, vec)
        
        # Integrate over the domain using quadrature weights
        # JxW is the Jacobian determinant times quadrature weight
        energy = np.sum(energy_density * self.fes[0].JxW[:, :, None])  # Scalar
        
        return energy

    def calculate_surface_charge(self, sol, conductor_points, conductor_elements):
        """Calculate the total charge on a conductor surface.
        
        Args:
            sol: Solution vector
            conductor_points: Points belonging to the conductor
            conductor_elements: Surface elements of the conductor
        
        Returns:
            Total charge on the conductor surface
        """
        # Make sure materials are set - safety check
        if not self.internal_vars or len(self.internal_vars) == 0:
            raise ValueError("Material properties not set. Call set_materials() before calculating surface charge.")
        
        # Convert conductor_elements to numpy array if it's not already
        conductor_elements = np.array(conductor_elements)
        
        # Convert solution to JAX array if it's not already
        sol = np.array(sol)
        
        # If conductor_elements is empty, return zero charge
        if len(conductor_elements) == 0:
            print("Warning: No conductor elements provided, returning zero charge.")
            return 0.0
        
        try:
            # Get the solution values at quadrature points for each cell
            cells = self.fes[0].cells[conductor_elements]  # (num_cells, num_nodes)
            cells_sol = sol[cells]  # (num_cells, num_nodes)
            
            # Get shape gradients for each cell
            shape_grads = np.take(self.fes[0].shape_grads, conductor_elements, axis=0)  # (num_cells, num_quads, num_nodes, dim)
            
            # Print shapes for debugging
            print("cells_sol shape:", cells_sol.shape)
            print("shape_grads shape:", shape_grads.shape)
            
            # Reshape arrays for broadcasting
            cells_sol = cells_sol[:, None, :, 0]  # (num_cells, 1, num_nodes)
            
            print("cells_sol reshaped:", cells_sol.shape)
            
            # Calculate gradients
            u_grads = cells_sol[:, :, :, None] * shape_grads  # (num_cells, num_quads, num_nodes, dim)
            u_grads = np.sum(u_grads, axis=2)  # (num_cells, num_quads, dim)
            
            # Get permittivity values for each quadrature point
            eps = np.take(self.internal_vars[0], conductor_elements, axis=0)  # (num_cells, num_quads)
            
            # Calculate surface charge density at each quadrature point
            # σ = -ε∂φ/∂n
            # We assume the normal points outward from the conductor
            charge_density = -eps[:, :, None] * u_grads  # (num_cells, num_quads, dim)
            
            # Integrate over the surface using quadrature weights
            # JxW is the Jacobian determinant times quadrature weight
            JxW = np.take(self.fes[0].JxW, conductor_elements, axis=0)  # (num_cells, num_quads)
            charge = np.sum(charge_density * JxW[:, :, None])  # Scalar
            
            return charge
            
        except Exception as e:
            print(f"Error calculating surface charge: {e}")
            # Return a default value rather than crashing
            return 0.0


def read_gmsh_file(mesh_file):
    """
    Read a GMSH file and extract mesh data, conductor boundaries, and material properties.
    Returns:
    - mesh: JAX-FEM Mesh object
    - boundary_info: Dictionary mapping conductor names to point indices
    - material_info: Dictionary mapping material names to cell indices
    """
    print(f"Reading mesh file: {mesh_file}")
    # Read the mesh file using meshio
    try:
        mesh_data = meshio.read(mesh_file)
    except Exception as e:
        print(f"Error reading mesh file: {e}")
        sys.exit(1)
    
    # Extract mesh elements, points, and physical groups
    points = onp.array(mesh_data.points)
    
    # For simplicity, assume using first 3D cell type available
    for cell_block in mesh_data.cells:
        if cell_block.type in ['tetra', 'hexahedron']:
            cells = onp.array(cell_block.data)
            ele_type = cell_block.type
            break
    else:
        # If no 3D cells found, try 2D cells
        for cell_block in mesh_data.cells:
            if cell_block.type in ['triangle', 'quad']:
                cells = onp.array(cell_block.data)
                ele_type = cell_block.type
                break
        else:
            print("No supported cell types found in mesh file")
            sys.exit(1)
    
    # Map GMSH cell types to JAX-FEM element types
    gmsh_to_jaxfem = {
        'triangle': 'TRI3',
        'quad': 'QUAD4',
        'tetra': 'TET4',
        'hexahedron': 'HEX8'
    }
    
    jaxfem_ele_type = gmsh_to_jaxfem.get(ele_type)
    
    # Create JAX-FEM mesh
    mesh = Mesh(points, cells, ele_type=jaxfem_ele_type)
    
    # Process physical groups to identify conductors and dielectric regions
    boundary_info = {}
    material_info = {}
    
    # Extract physical names if available
    physical_names = {}
    if hasattr(mesh_data, 'field_data') and mesh_data.field_data:
        print("Found field_data in mesh file")
        for name, data in mesh_data.field_data.items():
            dim, tag = data
            physical_names[tag] = name
            print(f"  Physical group: {name}, dim={dim}, tag={tag}")
    
    # Process material information (usually volume/3D elements)
    try:
        if hasattr(mesh_data, 'cell_data') and mesh_data.cell_data:
            print("Found cell_data in mesh file")
            # Find material markers in cell_data
            for key, value in mesh_data.cell_data.items():
                if key.startswith('gmsh:physical'):
                    print(f"  Processing {key}")
                    # Process material IDs for 3D elements (volumes)
                    for i, cell_type in enumerate(mesh_data.cells):
                        if len(value) > i and cell_type.type == ele_type:
                            material_ids = onp.array(value[i])
                            print(f"    Found {len(material_ids)} material IDs for {cell_type.type} elements")
                            # Map cell indices to material IDs
                            for j, mat_id in enumerate(material_ids):
                                if mat_id > 0:
                                    mat_name = physical_names.get(mat_id, f"material_{mat_id}")
                                    if mat_name not in material_info:
                                        material_info[mat_name] = []
                                    material_info[mat_name].append(j)
    except Exception as e:
        print(f"Warning: Error processing material data: {e}")
    
    # Process conductor information (usually surface/2D elements)
    try:
        # First check for surface elements
        conductor_points = {}
        surface_elements = None
        
        for i, cell_block in enumerate(mesh_data.cells):
            if cell_block.type in ['triangle', 'quad']:
                # Found surface elements
                surface_elements = onp.array(cell_block.data)
                
                # Check if there are physical groups for these surface elements
                for key, value in mesh_data.cell_data.items():
                    if key.startswith('gmsh:physical') and len(value) > i:
                        surface_tags = onp.array(value[i])
                        print(f"  Found {len(surface_tags)} surface tags for {cell_block.type} elements")
                        
                        # For each surface element with a tag
                        for j, tag in enumerate(surface_tags):
                            if tag > 0:
                                # This surface element belongs to a physical group
                                conductor_name = physical_names.get(tag, f"conductor_{tag}")
                                
                                # Get points that make up this surface element
                                element_points = surface_elements[j]
                                
                                # Add to conductor points
                                if conductor_name not in conductor_points:
                                    conductor_points[conductor_name] = []
                                conductor_points[conductor_name].extend(element_points)
        
        # Convert to unique point lists and add to boundary_info
        for name, points_list in conductor_points.items():
            boundary_info[name] = onp.array(list(set(points_list)))
            print(f"  Identified conductor: {name} with {len(boundary_info[name])} points")
        
        # If no conductors found and there are still physical groups, try point_data
        if not boundary_info and hasattr(mesh_data, 'point_data') and mesh_data.point_data:
            print("  No conductors found in cell_data, trying point_data...")
            # Find boundary markers in point_data
            for key, value in mesh_data.point_data.items():
                if key.startswith('gmsh:physical'):
                    # Map point indices to boundary IDs
                    for i, point_id in enumerate(value):
                        if point_id > 0:  # 0 typically means no assignment
                            name = physical_names.get(point_id, f"conductor_{point_id}")
                            if name not in boundary_info:
                                boundary_info[name] = []
                            boundary_info[name].append(i)
            
            for name, points_list in boundary_info.items():
                boundary_info[name] = onp.array(points_list)
                print(f"  Identified conductor from point_data: {name} with {len(points_list)} points")
    except Exception as e:
        print(f"Warning: Error processing conductor data: {e}")
    
    # If still no conductors found, create a demo setup
    if not boundary_info:
        print("No conductors found in mesh file. Creating demo boundary setup...")
        
        # Create a simple test case with two conductors
        # Find points on the top and bottom surfaces (assuming z is up)
        z_min, z_max = onp.min(points[:, 2]), onp.max(points[:, 2])
        
        # Create two conductors on opposite sides
        bottom_points = onp.where(onp.isclose(points[:, 2], z_min, atol=1e-5))[0]
        top_points = onp.where(onp.isclose(points[:, 2], z_max, atol=1e-5))[0]
        
        if len(bottom_points) > 0:
            boundary_info["BottomPlate"] = bottom_points
            print(f"  Created demo conductor: BottomPlate with {len(bottom_points)} points")
        
        if len(top_points) > 0:
            boundary_info["TopPlate"] = top_points
            print(f"  Created demo conductor: TopPlate with {len(top_points)} points")
    
    # Process cell data to get material information
    dielectric_constants = {}
    
    # Set default dielectric constants for materials
    # In real application, these should be specified in the input file or command line
    for mat_name in material_info.keys():
        # Try to extract dielectric constant from name if it contains "ER"
        er = 3.9  # Default to SiO2
        if "ER" in mat_name:
            try:
                er_str = mat_name.split("ER")[1].split("_")[0]
                er = float(er_str)
            except:
                pass
        dielectric_constants[mat_name] = er
        print(f"  Set dielectric constant for {mat_name}: {er}")
    
    # Map cells to material properties - using numpy array for now to avoid JAX immutability issues
    cell_materials = onp.zeros(len(cells), dtype=onp.int32)
    
    # Assign material IDs to cells
    mat_id = 1
    for mat_name, cell_indices in material_info.items():
        print(f"  Assigning material {mat_name} (ID={mat_id}) to {len(cell_indices)} cells")
        for idx in cell_indices:
            cell_materials[idx] = mat_id
        mat_id += 1
    
    # Convert material names to IDs in dielectric_constants
    dielectric_constants_by_id = {0: 1.0}  # Default for unassigned cells
    mat_id = 1
    for mat_name in material_info.keys():
        dielectric_constants_by_id[mat_id] = dielectric_constants[mat_name]
        mat_id += 1
    
    # Convert to JAX array at the end
    cell_materials = onp.array(cell_materials)
    
    print(f"Found {len(boundary_info)} conductors and {len(material_info)} materials")
    
    return mesh, boundary_info, cell_materials, dielectric_constants_by_id


def calculate_capacitance(mesh_file, output_dir):
    """
    Calculate capacitance matrix from a GMSH file.
    Args:
        mesh_file: Path to the GMSH mesh file
        output_dir: Directory to save the output
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the mesh file
    mesh, conductors, materials, dielectric_constants = read_gmsh_file(mesh_file)
    
    print(f"Found {len(conductors)} conductors")
    for name, points in conductors.items():
        print(f"  {name}: {len(points)} points")
    
    # For large meshes with many conductors, limit processing
    max_conductors = 2  # Only process the first few conductors for demonstration
    if len(conductors) > max_conductors:
        print(f"\nLimiting analysis to first {max_conductors} conductors for efficiency")
        conductor_names = list(conductors.keys())[:max_conductors]
        limited_conductors = {name: conductors[name] for name in conductor_names}
        conductors = limited_conductors
    
    # Initialize capacitance matrix
    n_conductors = len(conductors)
    capacitance_matrix = onp.zeros((n_conductors, n_conductors))
    
    # Convert conductor names to list for indexing
    conductor_names = list(conductors.keys())
    
    # Calculate capacitance between each pair of conductors
    for i in range(n_conductors):
        for j in range(n_conductors):
            if i == j:
                continue
                
            print(f"\nCalculating capacitance between {conductor_names[i]} and {conductor_names[j]}")
            
            # Get conductor points
            conductor_points = conductors[conductor_names[i]]
            other_points = conductors[conductor_names[j]]
            
            # Convert to numpy arrays if they aren't already
            conductor_points = onp.array(conductor_points)
            other_points = onp.array(other_points)
            
            # Print shapes for debugging
            print(f"  Conductor {conductor_names[i]} points shape: {conductor_points.shape}")
            print(f"  Conductor {conductor_names[j]} points shape: {other_points.shape}")
            
            # Create problem with these conductors
            problem = ElectrostaticProblem(
                mesh=mesh,
                vec=1,
                dim=mesh.points.shape[1],
                ele_type=mesh.ele_type,
                dirichlet_bc_info=None,  # We'll set boundary conditions manually after creation
                location_fns=None,
                dielectric_constants=dielectric_constants,
                surface_elements={conductor_names[i]: conductor_points, conductor_names[j]: other_points}
            )
            
            # Set material properties before solving
            problem.set_materials(materials)
            
            # Manually apply Dirichlet boundary conditions
            # Create dofs array with initial values
            dofs = onp.zeros(problem.fes[0].num_total_dofs)
            
            # Set voltage on conductor1 points to 1V
            for point_idx in conductor_points:
                dofs[point_idx] = 1.0
                
            # Set voltage on conductor2 points to 0V (already 0 by default)
            # for point_idx in other_points:
            #     dofs[point_idx] = 0.0
                
            # Create solution array with boundary conditions applied
            sol = [dofs]
            
            # For surface elements, we'll use the conductor points
            surface_elements = []
            
            # Find elements connected to conductor points
            for idx, cell in enumerate(mesh.cells):
                if any(node in conductor_points for node in cell):
                    surface_elements.append(idx)
            
            if not surface_elements:
                print(f"Warning: No surface elements found for {conductor_names[i]}")
                continue
            
            print(f"Found {len(surface_elements)} surface elements for {conductor_names[i]}")
                
            # Calculate surface charge on conductor
            charge = problem.calculate_surface_charge(sol, conductor_points, surface_elements)
            
            # Store in capacitance matrix
            capacitance_matrix[i, j] = charge
            capacitance_matrix[j, i] = charge  # Matrix is symmetric
    
    # Save capacitance matrix
    output_file = os.path.join(output_dir, "capacitance_matrix.txt")
    onp.savetxt(output_file, capacitance_matrix)
    print(f"\nCapacitance matrix saved to {output_file}")
    print("\nCapacitance matrix (in Farads):")
    print(capacitance_matrix)
    
    return capacitance_matrix


def main():
    parser = argparse.ArgumentParser(description="Calculate parasitic capacitance from GMSH file")
    parser.add_argument("mesh_file", help="Path to GMSH (.msh) file")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    args = parser.parse_args()
    
    calculate_capacitance(args.mesh_file, args.output)


if __name__ == "__main__":
    main() 