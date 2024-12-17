import os
import numpy as np

from data.asset_manager import StructureAssets
from mesh_generation.util import landmark_constriction_site, landmark_ptc

def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create a rotation matrix for rotating around an axis by an angle.
    Matches the implementation from the reference code.
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    R = (np.eye(3) + np.sin(angle) * K + 
         (1 - np.cos(angle)) * (K @ K))
    
    return R

def read_mmcif(filename):
    """Read mmCIF file and extract atom coordinates."""
    atoms = []
    header_lines = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            parts = line.strip().split()
            if len(parts) >= 14:
                x = float(parts[10])
                y = float(parts[11])
                z = float(parts[12])
                atoms.append({
                    'coords': np.array([x, y, z]),
                    'line': line,
                    'header': False
                })
        else:
            atoms.append({
                'coords': None,
                'line': line,
                'header': True
            })
    
    return atoms

def slice_mmcif_with_output(input_file: str,


                           base_point: np.ndarray,
                           axis_point: np.ndarray,
                           angle: float,
                           output_file: str):
    """
    Slice a mmCIF file using the same geometry as the reference PDB slicer.
    
    Parameters:
    input_file: str, path to input mmCIF file
    base_point: np.ndarray, starting point of the axis [x,y,z]
    axis_point: np.ndarray, end point of the axis [x,y,z]
    angle: float, angle in degrees for the slicing plane rotation
    output_file: str, path for output mmCIF file
    """
    # Read mmCIF file
    atoms = read_mmcif(input_file)
    
    # Extract coordinates for non-header atoms
    coords = np.array([atom['coords'] for atom in atoms if not atom['header']])
    
    # Calculate plane normal using the same method as reference code
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
    
    # Create mask for kept half (keeping negative distances like reference code)
    kept_mask = distances <= 0
    
    # Write output mmCIF file
    atom_count = 0
    with open(output_file, 'w') as f:
        coord_idx = 0
        for atom in atoms:
            if atom['header']:
                f.write(atom['line'])
            else:
                if kept_mask[coord_idx]:
                    # Update atom numbering in the line
                    parts = atom['line'].split()
                    parts[1] = str(atom_count + 1)  # Update atom number
                    f.write(' '.join(parts) + '\n')
                    atom_count += 1
                coord_idx += 1
    
    print(f"Wrote {sum(kept_mask)} atoms to {output_file}")
    return sum(kept_mask)

RCSB_ID     = "4UG0"
data_dir    = os.getenv("DATA_DIR")
if __name__ == "__main__":
    SA = StructureAssets(data_dir, RCSB_ID)
    base_point = np.array(landmark_ptc(RCSB_ID))
    axis_point = np.array(landmark_constriction_site(RCSB_ID))
    slice_mmcif_with_output(SA.cif_struct, base_point, axis_point, 45, SA.cif_struct.split('.cif')[0]+'_half.cif')