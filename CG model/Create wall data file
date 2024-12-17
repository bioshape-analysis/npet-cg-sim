import stl
from stl import mesh
import numpy as np
import trimesh
import aspose.threed as a3d
import pymeshlab as ml
import pymeshlab


# Some useful functions

def write_data_file(filename, num_atoms):
    """Create the header section of the wall particle data file for LAMMPS."""
    with open(filename, 'w') as f:
    
        # Write the initial section
        f.write("\n")
        f.write(f"{num_atoms}  atoms\n")
        f.write("0  bonds\n")
        f.write("0  angles\n\n")
        
        f.write("1  atom types\n")
        f.write("0  bond types\n")
        f.write("0  angle types\n\n")
        
        f.write("0.0000 1000.0000 xlo xhi\n")
        f.write("0.0000 1000.0000 ylo yhi\n")
        f.write("0.0000 1000.0000 zlo zhi\n\n")
        
        f.write("Masses\n\n")
        f.write("1    12\n\n")  
        
        f.write("Atoms\n\n")


def append_coordinates_to_file(output_filename, final_points):
    """
    Appends wall particle coordinates to a specified output file.

    Parameters:
    output_filename (str): The path to the output file.
    vertices (list): The list of tuples containing coordinates.
    """
    with open(output_filename, 'a') as file:  # Open in append mode
        for i, (x, y, z) in enumerate(final_points, start=1):

            # Write the formatted line
            file.write(f"{i} \t1\t1\t0.0\t{x:.6f}\t{y:.6f}\t{z:.6f}\n")


def parse_stl(stl_file):
    """Calculate the rge vertices, normals and centroids of each triangular mesh surface, also output the coordinates of the wall particles."""
    # Load the STL file
    mesh_data = mesh.Mesh.from_file(stl_file)

    # Extract vertices and faces
    vertices = mesh_data.vectors  # Keep vectors grouped as triangles
    normals = mesh_data.normals  # Extract normals

    # Compute centroids of each triangle
    centroids = np.mean(vertices, axis=1)

    # Compute the norm for each normal vector individually
    norm = np.linalg.norm(normals, axis=1, keepdims=True) 
    normalized_vectors = normals / norm  # Normalize the normals

    # Compute new points along the normal vectors at a distance of 1.75 from the centroids
    new_points = centroids + 1.75 * normalized_vectors

    return vertices, normals, new_points, centroids, normalized_vectors

    
def compute_vertex_normals_from_stl(stl_file):
    """
    Compute vertex normals from an STL file. (if you want to put the beads along the vertex normal of mesh)

    Parameters:
    - stl_file: Path to the STL file.

    Returns:
    - vertices: A NumPy array of unique vertices (n_vertices, 3).
    - vertex_normals: A NumPy array of normals corresponding to each vertex (n_vertices, 3).
    """
    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(stl_file)

    # Extract the triangle vertices and face normals
    triangles = stl_mesh.vectors  # Shape (n_triangles, 3, 3)
    face_normals = stl_mesh.normals  # Shape (n_triangles, 3)

    # Flatten the triangle vertices into a single array
    all_vertices = triangles.reshape(-1, 3)  # Shape (3 * n_triangles, 3)

    # Find unique vertices and their indices
    unique_vertices, indices = np.unique(all_vertices, axis=0, return_inverse=True)

    # Initialize an array to accumulate normals for each unique vertex
    vertex_normals = np.zeros_like(unique_vertices)

    # Accumulate normals for each vertex
    for i, normal in enumerate(face_normals):
        # Each triangle has 3 vertices; assign the face normal to all 3
        for j in range(3):
            vertex_index = indices[i * 3 + j]
            vertex_normals[vertex_index] += normal

    # Normalize the accumulated normals to make them unit vectors
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)

    # Avoid division by zero
    vertex_normals = vertex_normals / np.maximum(norms, 1e-8)  
    new_points = unique_vertices + 1.75 * vertex_normals

    return unique_vertices, vertex_normals, new_points


def compute_edge_normals(stl_file):
    """
    Compute midpoints and normals for mesh edges (if you want to put the beads along the normal vector of mesh edges)

    Parameters:
    - mesh_data: STL mesh data loaded using numpy-stl.
    - face_normals: Normal vectors for each face (n_triangles, 3).

    Returns:
    - edge_midpoints: Midpoints of edges (n_edges, 3).
    - edge_normals: Normals associated with each edge (n_edges, 3).
    """
    # Extract vertices of each triangle
    stl_mesh = mesh.Mesh.from_file(stl_file)

    face_normals = stl_mesh.normals 
    triangles = stl_mesh.vectors

    # Define edges for each triangle
    edges = [
        (0, 1),  # Edge from vertex 0 to 1
        (1, 2),  # Edge from vertex 1 to 2
        (2, 0),  # Edge from vertex 2 to 0
    ]

    # Initialize lists to store edge midpoints and normals
    edge_midpoints = []
    edge_normals = []

    for i, triangle in enumerate(triangles):
        normal = face_normals[i]
        for edge in edges:
            v1, v2 = triangle[edge[0]], triangle[edge[1]]
            
            # Compute midpoint of the edge
            midpoint = (v1 + v2) / 2  
            edge_midpoints.append(midpoint)

            # Use face normal for the edge
            edge_normals.append(normal)  

    # Convert lists to arrays
    edge_midpoints = np.array(edge_midpoints)
    edge_normals = np.array(edge_normals)

    # Normalize edge normals
    edge_normals /= np.linalg.norm(edge_normals, axis=1, keepdims=True)
    edge_new_points = edge_midpoints + 1.75 * edge_normals
    
    return edge_midpoints, edge_normals, edge_new_points


# Example Usage for tunnel wall particles
# Exactly the same processes for ribosome surface particles

molecule = "4UG0"

# Change .ply to .stl, make mesh structure readable in LAMMPS
scene = a3d.Scene.from_file(f"/Users/Desktop/{molecule}_tunnel.ply")
scene.save(f"/Users/Desktop/{molecule}_tunnel.stl")

vertices, normals, new_points, centroids, normalized_vector = parse_stl(f'/Users/Desktop/{molecule}_tunnel.stl')

# Append formatted coordinates to the output file
output_filename = f'/Users/Desktop/data.tunnel' 
write_data_file(output_filename,len(new_points))
append_coordinates_to_file(output_filename, new_points)


