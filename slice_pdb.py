import numpy as np
from Bio import PDB
from typing import Tuple, List, Union
import pyvista as pv

from mesh_generation.util import landmark_constriction_site, landmark_ptc


def slice_pdb_with_output(pdb_path: str,
                         base_point: np.ndarray,
                         axis_point: np.ndarray,
                         angle: float,
                         output_path: str):
    """
    Slice a PDB file and output the kept half as a new PDB file.
    Now keeps the opposite half (negative distances from plane).
    
    Parameters:
    pdb_path: str, path to input PDB file
    base_point: np.ndarray, starting point of the axis [x,y,z]
    axis_point: np.ndarray, end point of the axis [x,y,z]
    angle: float, angle in degrees for the slicing plane rotation
    output_path: str, path for output PDB file
    """
    # Read PDB file and store both coordinates and full lines
    coords = []
    pdb_lines = []
    crystal_line = ""
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('CRYST'):
                crystal_line = line
                continue
            elif line.startswith('HETATM'):
                # Extract coordinates
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
                pdb_lines.append(line)
    
    coords = np.array(coords)
    
    # Calculate plane normal
    axis = axis_point - base_point
    axis_unit = axis / np.linalg.norm(axis)
    
    if not np.allclose(axis_unit, [1, 0, 0]):
        init_perp = np.cross(axis_unit, [1, 0, 0])
    else:
        init_perp = np.cross(axis_unit, [0, 1, 0])
    init_perp = init_perp / np.linalg.norm(init_perp)
    
    # Rotate the initial perpendicular vector around the axis
    rotation_matrix = rotation_matrix_from_axis_angle(axis_unit, np.radians(angle))
    plane_normal = rotation_matrix @ init_perp
    
    # Calculate signed distances from points to the plane
    points_centered = coords - base_point
    distances = np.dot(points_centered, plane_normal)
    
    # Create mask for kept half - now keeping negative distances
    kept_mask = distances <= 0  # Changed from >= to <= to keep opposite half
    
    # Write output PDB file
    with open(output_path, 'w') as f:
        # Write crystal information
        f.write(crystal_line)
        
        # Write kept atoms with updated numbering
        atom_num = 1
        for i, keep in enumerate(kept_mask):
            if keep:
                # Get original line and update atom number
                line = pdb_lines[i]
                new_line = f"{line[:6]}{atom_num:5d}{line[11:]}"
                f.write(new_line)
                atom_num += 1
    
    print(f"Wrote {sum(kept_mask)} atoms to {output_path}")
    return sum(kept_mask)

def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create a rotation matrix for rotating around an axis by an angle.
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    R = (np.eye(3) + np.sin(angle) * K + 
         (1 - np.cos(angle)) * (K @ K))
    
    return R


RCSB_ID  = "4UG0"
struct = 'output_tunnel.pdb'
clipped= 'tunnel-clipped.pdb'
if __name__ == "__main__":
    base_point          = np.array(landmark_ptc(RCSB_ID))
    axis_point = np.array(landmark_constriction_site(RCSB_ID))
    
    angle = 45  # degrees
    
        # Visualize the slice
    n_atoms = slice_pdb_with_output(
        struct,
        base_point,
        axis_point,
        angle,
        clipped
    )