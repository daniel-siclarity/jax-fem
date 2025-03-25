#!/usr/bin/env python3

"""
Demo script for calculating parasitic capacitance using JAX-FEM.
This script creates a simple capacitor geometry using GMSH,
then calculates the capacitance using our application.
"""

import os
import sys
import numpy as np
import gmsh
import argparse

# Try different import methods to handle various execution contexts
try:
    # When running as a module (python -m applications.capacitance.demo)
    from applications.capacitance.capacitance import calculate_capacitance
except ImportError:
    try:
        # When running from the same directory
        from .capacitance import calculate_capacitance
    except ImportError:
        # Fallback for direct execution
        from capacitance import calculate_capacitance

def create_parallel_capacitor_mesh(output_dir, plate_size=1.0, plate_distance=0.2):
    """Create a parallel plate capacitor mesh using GMSH"""
    # Initialize GMSH
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    try:
        # Create a new model
        gmsh.model.add("parallel_capacitor")
        
        # Create geometry using built-in kernel
        # Create points for bottom plate
        p1 = gmsh.model.geo.addPoint(0, 0, 0)
        p2 = gmsh.model.geo.addPoint(plate_size, 0, 0)
        p3 = gmsh.model.geo.addPoint(plate_size, plate_size, 0)
        p4 = gmsh.model.geo.addPoint(0, plate_size, 0)
        
        # Create points for top plate
        p5 = gmsh.model.geo.addPoint(0, 0, plate_distance)
        p6 = gmsh.model.geo.addPoint(plate_size, 0, plate_distance)
        p7 = gmsh.model.geo.addPoint(plate_size, plate_size, plate_distance)
        p8 = gmsh.model.geo.addPoint(0, plate_size, plate_distance)
        
        # Create lines for bottom plate
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)
        
        # Create lines for top plate
        l5 = gmsh.model.geo.addLine(p5, p6)
        l6 = gmsh.model.geo.addLine(p6, p7)
        l7 = gmsh.model.geo.addLine(p7, p8)
        l8 = gmsh.model.geo.addLine(p8, p5)
        
        # Create vertical lines connecting plates
        l9 = gmsh.model.geo.addLine(p1, p5)
        l10 = gmsh.model.geo.addLine(p2, p6)
        l11 = gmsh.model.geo.addLine(p3, p7)
        l12 = gmsh.model.geo.addLine(p4, p8)
        
        # Create curve loops
        bottom_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        top_loop = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
        front_loop = gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])
        right_loop = gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])
        back_loop = gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])
        left_loop = gmsh.model.geo.addCurveLoop([l4, l9, -l8, -l12])
        
        # Create surfaces
        bottom_surface = gmsh.model.geo.addPlaneSurface([bottom_loop])
        top_surface = gmsh.model.geo.addPlaneSurface([top_loop])
        front_surface = gmsh.model.geo.addPlaneSurface([front_loop])
        right_surface = gmsh.model.geo.addPlaneSurface([right_loop])
        back_surface = gmsh.model.geo.addPlaneSurface([back_loop])
        left_surface = gmsh.model.geo.addPlaneSurface([left_loop])
        
        # Create surface loop and volume
        surface_loop = gmsh.model.geo.addSurfaceLoop([
            bottom_surface, top_surface,
            front_surface, right_surface,
            back_surface, left_surface
        ])
        volume = gmsh.model.geo.addVolume([surface_loop])
        
        # Synchronize the model
        gmsh.model.geo.synchronize()
        
        # Create physical groups
        bottom_plate = gmsh.model.addPhysicalGroup(2, [bottom_surface], tag=1)
        top_plate = gmsh.model.addPhysicalGroup(2, [top_surface], tag=2)
        dielectric = gmsh.model.addPhysicalGroup(3, [volume], tag=3)
        
        # Set names for the physical groups
        gmsh.model.setPhysicalName(2, bottom_plate, "BottomPlate")
        gmsh.model.setPhysicalName(2, top_plate, "TopPlate")
        gmsh.model.setPhysicalName(3, dielectric, "Dielectric_ER3.9")
        
        # Set mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", plate_size/10)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", plate_size/5)
        
        # Generate 3D mesh
        gmsh.model.mesh.generate(3)
        
        # Save mesh
        os.makedirs(output_dir, exist_ok=True)
        mesh_file = os.path.join(output_dir, "parallel_capacitor.msh")
        gmsh.write(mesh_file)
        
        return mesh_file
        
    finally:
        gmsh.finalize()

def create_interdigitated_capacitor_mesh(
    width=1.0,
    height=0.5,
    finger_width=0.1,
    finger_spacing=0.1,
    num_fingers=3,
    dielectric_er=3.9,
    mesh_size=0.02,
    output_file="interdigitated.msh"
):
    """
    Create an interdigitated capacitor mesh using GMSH.
    
    Args:
        width: Total width of the capacitor
        height: Height of the capacitor
        finger_width: Width of each finger
        finger_spacing: Spacing between fingers
        num_fingers: Number of fingers per electrode
        dielectric_er: Relative permittivity of the dielectric
        mesh_size: Size of mesh elements
        output_file: Path to save the mesh file
    
    Returns:
        Path to the created mesh file
    """
    # Initialize GMSH
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("interdigitated")
    
    # Create the dielectric substrate
    substrate = gmsh.model.occ.addBox(0, 0, 0, width, height, 0.1)
    
    # Create fingers for the first electrode
    electrode1_fingers = []
    for i in range(num_fingers):
        x_pos = i * (finger_width + finger_spacing)
        finger = gmsh.model.occ.addBox(
            x_pos, 0, 0.1,
            finger_width, height * 0.8, 0.01
        )
        electrode1_fingers.append(finger)
    
    # Create fingers for the second electrode
    electrode2_fingers = []
    for i in range(num_fingers - 1):
        x_pos = i * (finger_width + finger_spacing) + finger_width + finger_spacing/2
        finger = gmsh.model.occ.addBox(
            x_pos, height * 0.2, 0.1,
            finger_width, height * 0.8, 0.01
        )
        electrode2_fingers.append(finger)
    
    # Synchronize the model
    gmsh.model.occ.synchronize()
    
    # Add physical groups
    # Physical Volume for the dielectric
    dielectric_tag = gmsh.model.addPhysicalGroup(3, [substrate], name="Dielectric")
    gmsh.model.setPhysicalName(3, dielectric_tag, f"Dielectric_ER{dielectric_er}")
    
    # Physical Volume for electrode 1
    electrode1_tag = gmsh.model.addPhysicalGroup(3, electrode1_fingers, name="Electrode1")
    gmsh.model.setPhysicalName(3, electrode1_tag, "Electrode1")
    
    # Physical Volume for electrode 2
    electrode2_tag = gmsh.model.addPhysicalGroup(3, electrode2_fingers, name="Electrode2")
    gmsh.model.setPhysicalName(3, electrode2_tag, "Electrode2")
    
    # Set mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    
    # Generate 3D mesh
    gmsh.model.mesh.generate(3)
    
    # Save the mesh
    gmsh.write(output_file)
    
    # Finalize GMSH
    gmsh.finalize()
    
    print(f"Created mesh file: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Run capacitance demo")
    parser.add_argument("--type", choices=["parallel", "interdigitated"], default="parallel",
                      help="Type of capacitor to simulate")
    parser.add_argument("--output", "-o", default="./demo_output",
                      help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.type == "parallel":
        # Create parallel plate capacitor mesh
        mesh_file = create_parallel_capacitor_mesh(args.output)
        
        # Calculate capacitance
        calculate_capacitance(mesh_file, args.output)
    elif args.type == "interdigitated":
        # Create interdigitated capacitor mesh
        mesh_file = create_interdigitated_capacitor_mesh(
            width=1.0,
            height=0.5,
            finger_width=0.1,
            finger_spacing=0.1,
            num_fingers=3,
            dielectric_er=3.9,
            mesh_size=0.02,
            output_file=os.path.join(args.output, "interdigitated_capacitor.msh")
        )
        
        # Calculate capacitance
        calculate_capacitance(mesh_file, args.output)
    else:
        print(f"Capacitor type '{args.type}' not implemented yet")


if __name__ == "__main__":
    main() 