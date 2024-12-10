import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import plyfile
from mesh_generation.util import landmark_constriction_site, landmark_ptc

def clip_mesh_at_angle(mesh_path: str, 
                      base_point: np.ndarray, 
                      axis_point: np.ndarray, 
                      angle: float,
                      return_both: bool = False):
    """
    Clip a mesh along a plane defined by the axis and angle, returning one or both halves.
    
    Parameters:
    mesh_path: str, path to the input PLY file
    base_point: np.ndarray, starting point of the axis [x,y,z]
    axis_point: np.ndarray, end point of the axis [x,y,z]
    angle: float, angle in degrees for the clipping plane rotation around the axis
    return_both: bool, whether to return both halves of the mesh
    
    Returns:
    tuple: (clipped_mesh, other_half) if return_both=True, otherwise just clipped_mesh
    """
    # Load the mesh
    if isinstance(mesh_path, str):
        mesh = pv.read(mesh_path)
    else:
        mesh = mesh_path
    
    # Calculate axis vector and normalize it
    axis = axis_point - base_point
    axis_unit = axis / np.linalg.norm(axis)
    
    # Find a vector perpendicular to the axis for initial normal
    if not np.allclose(axis_unit, [1, 0, 0]):
        init_perp = np.cross(axis_unit, [1, 0, 0])
    else:
        init_perp = np.cross(axis_unit, [0, 1, 0])
    init_perp = init_perp / np.linalg.norm(init_perp)
    
    # Rotate the initial perpendicular vector around the axis
    rotation_matrix = rotation_matrix_from_axis_angle(axis_unit, np.radians(angle))
    normal = rotation_matrix @ init_perp
    
    # Clip the mesh
    clipped = mesh.clip(normal=normal, origin=base_point)
    
    if return_both:
        # Get the other half by inverting the normal
        other_half = mesh.clip(normal=-normal, origin=base_point)
        return clipped, other_half
    
    return clipped


def visualize_clipped_mesh(mesh_path: str,
                          base_point: np.ndarray,
                          axis_point: np.ndarray,
                          angle: float):
    """
    Visualize the clipping result with both halves in different colors.
    """
    # Get both halves of the mesh
    half1, half2 = clip_mesh_at_angle(mesh_path, base_point, axis_point, angle, return_both=True)
    
    # Create plotter
    p = pv.Plotter()
    
    # Add both halves with different colors
    p.add_mesh(half1, color='red', opacity=0.7, label='Kept Half')
    p.add_mesh(half2, color='blue', opacity=0.3, label='Clipped Half')
    
    # Add axis line
    line = pv.Line(base_point, axis_point)
    p.add_mesh(line, color='black', line_width=5)
    
    # Add points marking the axis
    points = pv.PolyData(np.vstack([base_point, axis_point]))
    p.add_mesh(points, color='yellow', point_size=10, render_points_as_spheres=True)
    
    # Add text showing the angle
    mid_point = (base_point + axis_point) / 2
    p.add_point_labels([mid_point], [f"Angle: {angle}°"], 
                      font_size=20, 
                      text_color='black',
                      shape_opacity=0.3)
    
    p.add_legend()
    p.show()

# def save_clipped_mesh(mesh_path: str,
#                      base_point: np.ndarray,
#                      axis_point: np.ndarray,
#                      angle: float,
#                      output_path: str):
#     """
#     Save the clipped half of the mesh to a file.
#     """
#     clipped = clip_mesh_at_angle(mesh_path, base_point, axis_point, angle)
#     clipped.save(output_path)

#     data            = plyfile.PlyData.read(output_path)
#     data.text       = True
#     ascii_duplicate = output_path.split(".ply")[0] + "_ascii.ply"
#     data.write(ascii_duplicate)
#     print(f"Saved clipped mesh to {output_path}")

def clip_mesh_with_normal_correction(mesh_path: str, 
                                   base_point: np.ndarray, 
                                   axis_point: np.ndarray, 
                                   angle: float,
                                   output_path: str):
    """
    Clip mesh and ensure proper face orientations in the result.
    
    Parameters:
    mesh_path: str, path to the input PLY file
    base_point: np.ndarray, starting point of the axis [x,y,z]
    axis_point: np.ndarray, end point of the axis [x,y,z]
    angle: float, angle in degrees for the clipping plane rotation around the axis
    output_path: str, where to save the corrected mesh
    """
    # Load the mesh
    mesh = pv.read(mesh_path)
    
    # Calculate clipping plane normal
    axis = axis_point - base_point
    axis_unit = axis / np.linalg.norm(axis)
    
    if not np.allclose(axis_unit, [1, 0, 0]):
        init_perp = np.cross(axis_unit, [1, 0, 0])
    else:
        init_perp = np.cross(axis_unit, [0, 1, 0])
    init_perp = init_perp / np.linalg.norm(init_perp)
    
    # Rotate the initial perpendicular vector around the axis
    rotation_matrix = rotation_matrix_from_axis_angle(axis_unit, np.radians(angle))
    normal = rotation_matrix @ init_perp
    
    # Clip the mesh
    clipped = mesh.clip(normal=normal, origin=base_point)
    
    # Fix mesh orientation
    fixed_mesh = clipped.triangulate()  # Ensure mesh is triangulated
    fixed_mesh.compute_normals(inplace=True)  # Recompute normals
    
    # Make normals consistent
    fixed_mesh.flip_normals()  # Flip all normals if they're pointing inward
    
    # Optional: Fill holes in the mesh
    filled_mesh = fixed_mesh.fill_holes(hole_size=fixed_mesh.length/100)
    
    # Save the corrected mesh
    filled_mesh.save(output_path)

    data            = plyfile.PlyData.read(output_path)
    data.text       = True
    ascii_duplicate = output_path.split(".ply")[0] + "_ascii.ply"
    data.write(ascii_duplicate)
    print(f"Saved clipped mesh to {output_path}")
    
    return filled_mesh

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

def verify_mesh_orientation(mesh_path: str):
    """
    Load a mesh and print diagnostics about its orientation.
    """
    mesh = pv.read(mesh_path)
    print(f"Mesh statistics:")
    print(f"Number of points: {mesh.n_points}")
    print(f"Number of faces: {mesh.n_faces}")
    print(f"Is all triangles: {mesh.is_all_triangles}")
    print(f"Number of open edges: {mesh.n_open_edges}")
    print(f"Is manifold: {mesh.is_manifold}")
    
    # Compute and display normal consistency
    mesh.compute_normals(inplace=True)
    normals = mesh.point_normals
    avg_normal = np.mean(normals, axis=0)
    print(f"Average normal direction: {avg_normal}")
    
    return mesh



# RCSB_ID  = "4UG0"
# meshpath = "4UG0.PR.watertight.ply"
# clippedpath = "4UG0.PR.watertight-clipped.ply"

RCSB_ID  = "4UG0"
meshpath = "./data/4UG0/alpha_shape_watertight_4UG0.ply"
clippedpath = "alpha_shape_watertight_4UG0-clipped.ply"


if __name__ == "__main__":
    
    # Generate array of angles to visualize
    angles = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    base_point          = np.array(landmark_ptc(RCSB_ID))
    axis_point = np.array(landmark_constriction_site(RCSB_ID))
    
    angle = 45  # degrees
    
    # Visualize the clipping
    visualize_clipped_mesh(meshpath, base_point, axis_point, angle)


    fixed_mesh = clip_mesh_with_normal_correction(
        meshpath,
        base_point,
        axis_point,
        45,
        clippedpath
    )

    # Verify the result
    verify_mesh_orientation(clippedpath)
    
    # # Save the clipped mesh
    # save_clipped_mesh(
    #     meshpath,
    #     base_point,
    #     axis_point,
    #     angle,
    #     clippedpath
    # )