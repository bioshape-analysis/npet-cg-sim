import os
import numpy as np
from typing import List, TextIO

import numpy as np
def get_transformation_to_C0(base_point: np.ndarray, axis_point: np.ndarray):
    """Get transformation matrices to align cylinder with z-axis."""
    # Get cylinder axis vector
    axis = axis_point - base_point
    axis_length = np.linalg.norm(axis)
    axis_unit = axis / axis_length

    # Get rotation that aligns axis_unit with [0, 0, 1]
    z_axis = np.array([0, 0, 1])

    if np.allclose(axis_unit, z_axis):
        R = np.eye(3)
    elif np.allclose(axis_unit, -z_axis):
        R = np.diag([1, 1, -1])
    else:
        v = np.cross(axis_unit, z_axis)
        s = np.linalg.norm(v)
        c = np.dot(axis_unit, z_axis)
        v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + v_skew + (v_skew @ v_skew) * (1 - c) / (s * s)

    return -base_point, R

def filter_points_in_cylinder(points: np.ndarray, ptc_pt: np.ndarray, 
                            constriction_pt: np.ndarray, radius: float, height: float):
    """Filter points to keep only those inside the cylinder."""
    # Transform points to align cylinder with z-axis
    translation, rotation = get_transformation_to_C0(ptc_pt, constriction_pt)
    points_translated = points + translation
    points_transformed = points_translated @ rotation.T
    
    # Filter points based on radius and height in transformed space
    xy_coords = points_transformed[:, :2]
    z_coords = points_transformed[:, 2]
    
    # Points must be:
    # 1. Within radius from z-axis (using xy coordinates)
    # 2. Between 0 and height in z coordinate
    radial_distances = np.linalg.norm(xy_coords, axis=1)
    mask = (radial_distances <= radius) & (z_coords >= 0) & (z_coords <= height)
    
    return points[mask]

def write_pointcloud_to_pdb(points: np.ndarray, output_file: str, 
                          ptc_pt: np.ndarray, constriction_pt: np.ndarray, 
                          radius: float, height: float):
    """
    Filter points by cylinder and write to PDB format.
    
    Args:
        points: Nx3 numpy array of point coordinates
        output_file: Path to output PDB file
        ptc_pt: Base point of cylinder (3D coordinates)
        constriction_pt: Top point of cylinder (3D coordinates)
        radius: Radius of cylinder
        height: Height of cylinder
    """
    # Filter points
    filtered_points = filter_points_in_cylinder(points, ptc_pt, constriction_pt, radius, height)
    
    print(f"Kept {len(filtered_points)}/{len(points)} points after cylinder filtering")
    
    with open(output_file, 'w') as f:
        f.write("CRYST1 1000.000 1000.000 1000.000  90.00  90.00  90.00 P 1           1\n")
        
        for i, (x, y, z) in enumerate(filtered_points, 1):
            f.write(f"HETATM{i:>5}  C   MOL {i:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          C\n")
        
        f.write("END\n")
    
    print(f"Wrote {output_file}")

def get_transformation_to_C0(base_point: np.ndarray, axis_point: np.ndarray):
    """Get transformation matrices to align cylinder with z-axis."""
    axis = axis_point - base_point
    axis_length = np.linalg.norm(axis)
    axis_unit = axis / axis_length

    z_axis = np.array([0, 0, 1])
    
    if np.allclose(axis_unit, z_axis):
        R = np.eye(3)
    elif np.allclose(axis_unit, -z_axis):
        R = np.diag([1, 1, -1])
    else:
        v = np.cross(axis_unit, z_axis)
        s = np.linalg.norm(v)
        c = np.dot(axis_unit, z_axis)
        v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + v_skew + (v_skew @ v_skew) * (1 - c) / (s * s)

    return -base_point, R

def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix for rotating around an axis by an angle."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], 
                  [axis[2], 0, -axis[0]], 
                  [-axis[1], axis[0], 0]])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R

def filter_points_half_cylinder(points: np.ndarray, ptc_pt: np.ndarray, 
                              constriction_pt: np.ndarray, radius: float, 
                              height: float, angle: float):
    """Filter points to keep only those inside half of the cylinder."""
    # First transform points to align cylinder with z-axis
    translation, rotation = get_transformation_to_C0(ptc_pt, constriction_pt)
    points_translated = points + translation
    points_transformed = points_translated @ rotation.T
    
    # Filter by cylinder bounds first
    xy_coords = points_transformed[:, :2]
    z_coords = points_transformed[:, 2]
    
    radial_distances = np.linalg.norm(xy_coords, axis=1)
    cylinder_mask = (radial_distances <= radius) & (z_coords >= 0) & (z_coords <= height)
    
    # Get points inside cylinder
    cylinder_points = points[cylinder_mask]
    
    # Now calculate clipping plane normal (similar to mesh clipping code)
    axis = constriction_pt - ptc_pt
    axis_unit = axis / np.linalg.norm(axis)
    
    if not np.allclose(axis_unit, [1, 0, 0]):
        init_perp = np.cross(axis_unit, [1, 0, 0])
    else:
        init_perp = np.cross(axis_unit, [0, 1, 0])
    init_perp = init_perp / np.linalg.norm(init_perp)
    
    # Rotate the initial perpendicular vector around the axis
    rot_matrix = rotation_matrix_from_axis_angle(axis_unit, np.radians(angle))
    normal = rot_matrix @ init_perp
    
    # Filter points based on which side of the plane they're on
    # Using dot product with normal to determine side
    vectors_to_points = cylinder_points - ptc_pt
    side_of_plane = np.dot(vectors_to_points, normal)
    half_mask = side_of_plane <0
    
    return cylinder_points[half_mask]

def write_pointcloud_to_mmcif(points: np.ndarray, output_file: str):
    """Write filtered points to minimal mmCIF format."""
    with open(output_file, 'w') as f:
        f.write("data_points\n")
        f.write("#\n")
        f.write("loop_\n")
        f.write("_atom_site.id\n")
        f.write("_atom_site.type_symbol\n")
        f.write("_atom_site.Cartn_x\n")
        f.write("_atom_site.Cartn_y\n")
        f.write("_atom_site.Cartn_z\n")
        
        for i, (x, y, z) in enumerate(points, 1):
            f.write(f"{i} C {x:.3f} {y:.3f} {z:.3f}\n")

def convert_pointcloud_to_halfslice_mmcif(points: np.ndarray, output_file: str,
                                        ptc_pt: np.ndarray, constriction_pt: np.ndarray,
                                        radius: float, height: float, angle: float):
    """Main function to filter points and convert to mmCIF."""
    filtered_points = filter_points_half_cylinder(points, ptc_pt, constriction_pt,
                                                radius, height, angle)
    
    print(f"Kept {len(filtered_points)}/{len(points)} points after filtering")
    write_pointcloud_to_mmcif(filtered_points, output_file)
    print(f"Wrote filtered points to {output_file}")

# Example usage:
if __name__ == "__main__":
    # Example data (replace with actual values)
    R = 40
    H = 100
    angle = 45  # degrees
    ptc_pt = np.array([0, 0, 0])
    constriction_pt = np.array([0, 0, 100])
    points = np.random.rand(1000, 3) * 200 - 100
    
    convert_pointcloud_to_halfslice_mmcif(points, "half_cylinder_points.cif",
                                        ptc_pt, constriction_pt, R, H, angle)