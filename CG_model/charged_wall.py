import re
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

proportionality = 0.5 # Proportionality constant for effective charge calculation, needs to be estimated

# Replace the following
molecule = '4UG0'
path = '/Path/To/Data/'  # Path to data files, needs to be set

# Parameters
epsilon_0 = 8.854*10**-12 #F/m
epsilon_rel = 7  #average value of protein, rna, and water 
lamda_d = 7.14

# Some useful functions

def potential(distance,charge):
    """Screened Coulomb potential calculations."""
    screened = np.exp(-distance/lamda_d)
    V_charge = screened*charge*1.6*10**-19*10**10/(4*np.pi*epsilon_0*epsilon_rel*distance)

    return V_charge

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two 3D points."""
    p1 = np.array(point1)
    p2 = np.array(point2)
    distance = np.linalg.norm(p2 - p1)

    return distance

def parse_opendx(file_path):
    """Parse OpenDX file obatined from APBS in PyMOL and extract grid coordinates and data values."""
    grid_counts = None
    origin = None
    deltas = []
    data = []
    in_data_section = False

    with open(file_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()

            # Extract grid dimensions
            if stripped_line.startswith("object 1 class gridpositions counts"):
                grid_counts = tuple(map(int, stripped_line.split()[-3:]))

            # Extract origin
            elif stripped_line.startswith("origin"):
                origin = np.array(list(map(float, stripped_line.split()[1:])))

            # Extract delta vectors
            elif stripped_line.startswith("delta"):
                deltas.append(np.array(list(map(float, stripped_line.split()[1:]))))

            # Detect the start of the data section
            elif stripped_line.startswith("object 3 class array"):
                in_data_section = True
            elif in_data_section and stripped_line and not stripped_line.startswith("attribute"):

                # Collect numerical data
                try:
                    data.extend(map(float, stripped_line.split()))
                except ValueError:
                    # Ignore any non-numerical lines
                    continue

    if not (grid_counts and origin is not None and deltas):
        raise ValueError("Failed to parse grid positions, origin, or deltas from the file.")

    # Create grid positions
    grids = [
        np.linspace(origin[i], origin[i] + (grid_counts[i] - 1) * deltas[i][i], grid_counts[i])
        for i in range(len(grid_counts))
    ]
    grid_positions = np.array(np.meshgrid(*grids, indexing='ij')).reshape(len(grid_counts), -1).T

    # Reshape data to match grid positions
    data_array = np.array(data).flatten()
    if len(data_array) != len(grid_positions):
        raise ValueError("Data array length does not match grid positions.")
    
    # Match coordinates to data
    matched_data = {tuple(coord): value for coord, value in zip(grid_positions, data_array)}

    return grid_positions, data_array, matched_data


def interpolate_spline(grid_origin, grid_deltas, grid_counts, data, query_points):
    """
    Perform spline interpolation to find values at non-grid points.

    Parameters:
    - grid_origin: List or array of the grid origin coordinates.
    - grid_deltas: List or array of delta vectors for each dimension.
    - grid_counts: Tuple of grid counts for each dimension.
    - data: Flattened 1D data array matching the grid.
    - query_points: List of points where interpolation is desired.

    Returns:
    - Interpolated values at the query points.
    """
    # Create the grid axes
    axes = [
        np.linspace(grid_origin[i], grid_origin[i] + (grid_counts[i] - 1) * grid_deltas[i][i], grid_counts[i])
        for i in range(len(grid_counts))
    ]
    
    # Reshape the data to match the grid
    data_reshaped = np.array(data).reshape(grid_counts)
    
    # Create the interpolator
    interpolator = RegularGridInterpolator(axes, data_reshaped, method='linear', bounds_error=False, fill_value=None)
    
    # Interpolate at the desired points
    interpolated_values = interpolator(query_points)

    return interpolated_values

def parse_dx_file(dx_file):
    """
    Extract grid info from dx.file
    """
    grid_origin = None
    grid_deltas = []
    grid_counts = None

    with open(dx_file, 'r') as file:
        for line in file:
            # Extract grid counts
            if "counts" in line:
                parts = line.split()
                grid_counts = tuple(map(int, parts[-3:]))  # Last 3 elements are counts

            # Extract grid origin
            elif line.startswith("origin"):
                parts = line.split()
                grid_origin = [float(parts[1]), float(parts[2]), float(parts[3])]

            # Extract grid deltas
            elif line.startswith("delta"):
                parts = line.split()
                grid_deltas.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return grid_origin, grid_deltas, grid_counts
    
# Functions for charge comaprison

def calculate_centerline_potentials(centerline_coords, charge_coords, charges, 
                                    lamda_d, epsilon_0, epsilon_rel):
    """
    Calculate electrostatic potential at each centerline point due to charged particles.
    
    Parameters:
    -----------
    centerline_coords : array-like, shape (n_points, 3)
        Coordinates of centerline points
    charge_coords : array-like, shape (n_charges, 3)
        Coordinates of charged particles (atoms or beads)
    charges : array-like, shape (n_charges,)
        Charge values for each particle
    max_distance : float
        Maximum distance to consider for potential calculations (Angstroms)
    
    Returns:
    --------
    potentials : array, shape (n_points,)
        Electrostatic potential at each centerline point
    """
    centerline_coords = np.array(centerline_coords)
    charge_coords = np.array(charge_coords)
    charges = np.array(charges)
    
    n_centerline_points = len(centerline_coords)
    potentials = np.zeros(n_centerline_points)
    
    # Calculate distances between all centerline points and all charged particles
    distances = cdist(centerline_coords, charge_coords, metric='euclidean')
    
    for i in range(n_centerline_points):
        q = 0
        for j in range(len(charges)):
            dist = distances[i,j]
            pot = potential(dist, charges[j], lamda_d, epsilon_0, epsilon_rel)
            q += pot

        potentials[i] = q
        
    return potentials

def fit_straight_line_and_project(centerline_coords):
    """
    Fit a straight line between the first and last centerline points,
    then project all centerline points onto this line.
    
    Parameters:
    -----------
    centerline_coords : array-like, shape (n_points, 3)
        Coordinates of centerline points
    
    Returns:
    --------
    distances : array, shape (n_points,)
        Distance along the straight line from start to each projected point
    line_start : array, shape (3,)
        Start point of the fitted line
    line_direction : array, shape (3,)
        Unit direction vector of the line
    """
    centerline_coords = np.array(centerline_coords)
    
    # Define straight line from first to last point
    line_start = centerline_coords[0]
    line_end = centerline_coords[-1]
    line_vector = line_end - line_start
    line_length = np.linalg.norm(line_vector)
    line_direction = line_vector / line_length  # Unit vector
    
    # Project each centerline point onto the straight line
    distances = np.zeros(len(centerline_coords))
    for i, point in enumerate(centerline_coords):
        # Vector from line start to current point
        point_vector = point - line_start
        # Project onto line direction (dot product gives distance along line)
        distances[i] = np.dot(point_vector, line_direction)
    
    return distances, line_start, line_direction

def compare_charge_distributions(centerline_coords, atom_coords, atom_charges, 
                               bead_coords, bead_charges, lamda_d, epsilon_0, epsilon_rel):
    """
    Compare atomistic and coarse-grained charge distributions along centerline.
    Calculates potentials at true centerline coordinates, then projects onto straight line.
    
    Parameters:
    -----------
    centerline_coords : array-like, shape (500, 3)
        Coordinates of 500 centerline points
    atom_coords : array-like, shape (n_atoms, 3)
        Coordinates of charged atoms near centerline
    atom_charges : array-like, shape (n_atoms,)
        Charges of atoms from PyMOL
    bead_coords : array-like, shape (n_beads, 3)
        Coordinates of coarse-grained beads
    bead_charges : array-like, shape (n_beads,)
        Charges of coarse-grained beads
    max_distance : float
        Maximum distance for potential calculations (Angstroms)
    
    Returns:
    --------
    centerline_distance : array
        Distance parameter along straight line projection
    atomistic_potentials : array
        Potential profile from atomistic charges
    cg_potentials : array
        Potential profile from coarse-grained charges
    """
    
    atomistic_potentials = calculate_centerline_potentials(
        centerline_coords, atom_coords, atom_charges, lamda_d, epsilon_0, epsilon_rel
    )
    
    cg_potentials = calculate_centerline_potentials(
        centerline_coords, bead_coords, bead_charges, lamda_d, epsilon_0, epsilon_rel
    )
    
    centerline_distance, line_start, line_direction = fit_straight_line_and_project(centerline_coords)
    
    return centerline_distance, atomistic_potentials, cg_potentials


def plot_comparison(centerline_distance, atomistic_potentials, cg_potentials, proportionality,
                   save_path=None):
    """
    Create comparison plot of atomistic vs coarse-grained potentials.
    
    Parameters:
    -----------
    centerline_distance : array
        Distance parameter along centerline (Angstroms)
    atomistic_potentials : array
        Potential profile from atomistic charges
    cg_potentials : array
        Potential profile from coarse-grained charges
    save_path : str, optional
        Path to save the plot
    """
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(centerline_distance, atomistic_potentials, 'b-', 
             linewidth=2, label='Atomistic (PyMOL)', alpha=0.8)
    plt.plot(centerline_distance, cg_potentials, 'r--', 
             linewidth=2, label='Coarse-Grained', alpha=0.8)
    
    plt.xlabel('Distance along centerline (Å)', fontsize=12)
    plt.ylabel('Electrostatic Potential (mV)', fontsize=12)
    plt.title(f'Comparison of Charge Distributions Along Centerline (prop = {proportionality})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    correlation = np.corrcoef(atomistic_potentials, cg_potentials)[0, 1]
    rmse = np.sqrt(np.mean((atomistic_potentials - cg_potentials)**2))
    
    textstr = f'Correlation: {correlation:.3f}\nRMSE: {rmse:.2e} mV'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    

def get_charged_atoms(pos_coordinates, neg_coordinates, pho_coordinates):
    """
    Collect charged atom coordinates.
    
    Parameters:
    -----------
    pos_coordinates : list of 3D coordinates of positively charged atoms
    neg_coordinates : list of 3D coordinates of negatively charged atoms
    pho_coordinates : list of 3D coordinates of phosphate groups
    
    Returns:
    --------
    all_atom : array of shape (n_atoms, 3)
        Combined coordinates of all charged atoms
    charges : array of shape (n_atoms,)
        Charge values for each atom (1 for positive, -1 for negative)
    """
    all_atom = np.concatenate((pos_coordinates, neg_coordinates, pho_coordinates))
    
    charges = np.ones(all_atom.shape[0])
    
    charges[len(pos_coordinates):] = -1 * np.ones(all_atom.shape[0] - len(pos_coordinates))
    
    return all_atom, charges


# 1. Example usage (for tunnel wall particles only)

# Store the coordinates of positively charged resdiues within 30 angstrom from the tunnel centerline derived from PyMOL
pos_coordinates = []

# Read and process the file
with open(f'{path}/30_pos_{molecule}.txt', 'r') as file:
    coordinates = []
    for line in file:
        cleaned_line = re.sub(r"\\", "", line).strip()
        coords = tuple(map(float, cleaned_line.split()))
        coordinates.append(coords)

# Remove duplicate coordinates
pos_coordinates = list(map(list, set(coordinates)))

# Store the coordinates of negatively charged resdiues within 30 angstrom from the tunnel centerline derived from PyMOL
neg_coordinates = []

# Read and process the file
with open(f'{path}/30_neg_{molecule}.txt', 'r') as file:
    coordinates = []
    for line in file:
        cleaned_line = re.sub(r"\\", "", line).strip()
        coords = tuple(map(float, cleaned_line.split()))
        coordinates.append(coords)

# Remove duplicate coordinates
neg_coordinates = list(map(list, set(coordinates)))
print(len(neg_coordinates),neg_coordinates[0])

# Store the coordinates of negatively charged phosphorus atoms within 30 angstrom from the tunnel centerline derived from PyMOL
pho_coordinates = []

# Read and process the file
with open(f'{path}/30_phos_{molecule}.txt', 'r') as file:
    coordinates = []
    for line in file:
        cleaned_line = re.sub(r"\\", "", line).strip()
        coords = tuple(map(float, cleaned_line.split()))
        coordinates.append(coords)

# Remove duplicate coordinates
pho_coordinates = list(map(list, set(coordinates)))
print(len(pho_coordinates),pho_coordinates[0])

################################### Read refined lammps tunnel data file ########################################

# Read the tunnel wall particle coordinates (new data file generated by LAMMPS after removing overlapping particles)
wall_coordinates = []
atom_ids = []

# Define flags to track the target section
in_target_section = False

# Read the file line by line
with open(f"{path}/data.{molecule}_tunnel_new", "r") as file:
    for line in file:

        # Strip whitespace from the beginning and end
        line = line.strip()
        
        # Check for the start and end of the target section
        if line.startswith("Atoms # full"):
            in_target_section = True  # Start reading after this line
            continue
        elif line.startswith("Velocities"):
            in_target_section = False  # Stop reading at this line
            break

        # If in the target section, process each line
        if in_target_section and line:
            # Split the line into columns
            columns = line.split()
            
            # Store the first column as an integer in `atom_ids`
            atom_ids.append(int(columns[0]))
            
            # Store the 5th, 6th, and 7th columns as 3D coordinates in `coordinates`
            coord = (float(columns[4]), float(columns[5]), float(columns[6]))
            wall_coordinates.append(coord)

# Output results to verify
print("Atom IDs:", len(atom_ids), atom_ids[0])
print("3D Coordinates:", len(wall_coordinates), wall_coordinates[0])

################################### calculate potentials for wall beads ########################################

# Calculate the effective Coulomb potential at each wall particle
V_wall = []
for i in range (len(wall_coordinates)):
    pot = 0
    for j1 in range (len(pos_coordinates)):
        distance = calculate_distance(wall_coordinates[i], pos_coordinates[j1])  
        V_charge = potential(distance,1)
        pot += V_charge
    for j2 in range (len(neg_coordinates)):
        distance = calculate_distance(wall_coordinates[i], neg_coordinates[j2])
        V_charge = potential(distance,-1)
        pot += V_charge
    for j3 in range (len(pho_coordinates)):
        distance = calculate_distance(wall_coordinates[i], pho_coordinates[j3])
        V_charge = potential(distance,-1)
        pot += V_charge
    V_wall.append(pot)

# Assign effective charges to each tunnel wall particle according to the effective Coulomb potentials
q_effect = []
for i in range (len(V_wall)):
    q_e = V_wall[i]*proportionality # Assumption for tunnel wall particles only, need to assign a proportionality value above
    q_effect.append(q_e)


# Generate a new data file for tunnel wall particles with effective charges
atom_charge_map = dict(zip(atom_ids, q_effect))

################### write new tunnel data lammps file including effective charges #######################

output_lines = []

# Flags to track the section of the file
in_target_section = False

# Open the file to read and the output file to write
with open(f"{path}/data.{molecule}_tunnel_new", "r") as infile, open(f"{path}/data.{molecule}_tunnel_charged", "w") as outfile:
    for line in infile:
        stripped_line = line.strip()

        # Write lines before the "Atoms # full" section
        if not in_target_section and stripped_line.startswith("Atoms # full"):
            in_target_section = True
            output_lines.append(line)  # Keep the "Atoms # full" line
            continue
        
        if not in_target_section:
            output_lines.append(line)  # Append lines before "Atoms # full"
            continue

        # Check for the end of the target section
        if stripped_line.startswith("Velocities"):
            in_target_section = False  # Stop reading at this line
            break

        # Process lines only in the target section
        if in_target_section and stripped_line:

            # Split line into columns
            columns = stripped_line.split()
            
            # Remove the last three columns
            columns = columns[:-3]

            # Convert the 1st column to an integer and check if it's in listA
            atom_id = int(columns[0])

            if atom_id in atom_charge_map:
                columns[3] = f"{atom_charge_map[atom_id]:.4f}" # Set the 4th column to '1' if atom_id is in listA
            # Rejoin the modified line and add it to the output list
            output_lines.append(" ".join(columns) + "\n")

    # Write all lines up to 'Velocity' to the output file
    outfile.writelines(output_lines)

print(f"File has been processed and saved as 'data.{molecule}_tunnel_charged', need futher modification")

# Be aware that the new data file need to be mannually modifed a bit. 
# The atom type should be changed to 1 and a blank line need to be added after the line "Atoms # full"


# 2. Example usage (for ribosome surface wall particles only)

# Read the ribsome surface particle coordinates (new data file generated by LAMMPS after removing overlapping particles)
sphere_coordinates = []
atom_ids = []

# Define flags to track the target section
in_target_section = False

# Read the file line by line
with open(f"{path}/data.{molecule}_sphere_new", "r") as file:
    for line in file:
        # Strip whitespace from the beginning and end
        line = line.strip()
        
        # Check for the start and end of the target section
        if line.startswith("Atoms # full"):
            in_target_section = True  # Start reading after this line
            continue
        elif line.startswith("Velocities"):
            in_target_section = False  # Stop reading at this line
            break

        # If in the target section, process each line
        if in_target_section and line:
            # Split the line into columns
            columns = line.split()
            
            # Store the first column as an integer in `atom_ids`
            atom_ids.append(int(columns[0]))
            
            # Store the 5th, 6th, and 7th columns as 3D coordinates in `coordinates`
            coord = (float(columns[4]), float(columns[5]), float(columns[6]))
            sphere_coordinates.append(coord)

# Output results to verify
print("Atom IDs:", len(atom_ids), atom_ids[0])
print("3D Coordinates:", len(sphere_coordinates),sphere_coordinates[0])

# Read the calculated potentials along the ribosome surface obtained from APBS
grid_positions, data_array, matched_data = parse_opendx(f"{path}/{molecule}_map.dx")

# Calculate the interpolated potentials at the ribosome surface wall particle coordinates
grid_origin, grid_deltas, grid_counts = parse_dx_file(f"{path}/{molecule}_map.dx")
data = data_array  
query_points = sphere_coordinates

interpolated_values = interpolate_spline(grid_origin, grid_deltas, grid_counts, data, query_points)*0.0256 # Change the units from kT/e to V

# Assign the effective charges to the ribosome surface wall particles according to interpolated potentials
V_sphere = interpolated_values
q_sphere = []
for i in range (len(V_sphere)):
    q_e = V_sphere[i]*proportionality # Uses same proportionality as for tunnel
    q_sphere.append(q_e)


# Generate a new data file for ribosome surface wall particles with effective charges
atom_charge_map = dict(zip(atom_ids, q_sphere))

output_lines = []

# Flags to track the section of the file
in_target_section = False

# Open the file to read and the output file to write
with open(f"{path}/data.{molecule}_sphere_new", "r") as infile, open(f"{path}/data.{molecule}_sphere_charged", "w") as outfile:
    for line in infile:
        stripped_line = line.strip()

        # Write lines before the "Atoms # full" section
        if not in_target_section and stripped_line.startswith("Atoms # full"):
            in_target_section = True
            output_lines.append(line)  # Keep the "Atoms # full" line
            continue
        
        if not in_target_section:
            output_lines.append(line)  # Append lines before "Atoms # full"
            continue

        # Check for the end of the target section
        if stripped_line.startswith("Velocities"):
            in_target_section = False  # Stop reading at this line
            #output_lines.append(line)  # Add the "Velocity" line to the output and stop processing
            break

        # Process lines only in the target section
        if in_target_section and stripped_line:
            # Split line into columns
            columns = stripped_line.split()
            
            # Remove the last three columns
            columns = columns[:-3]

            columns[1] = '1'
            columns[2] = '1'

            # Convert the 1st column to an integer and check if it's in listA
            atom_id = int(columns[0])

            if atom_id in atom_charge_map:
                #columns[1] = 1
                #columns[2] = 1
                columns[3] = f"{atom_charge_map[atom_id]:.4f}" # Set the 4th column to '1' if atom_id is in listA

            # Rejoin the modified line and add it to the output list
            output_lines.append(" ".join(columns) + "\n")

    # Write all lines up to 'Velocity' to the output file
    outfile.writelines(output_lines)

print(f"File has been processed and saved as 'data.{molecule}_sphere_charged', need futher modification")
# Be aware that the new data file need to be mannually modifed a bit. 
# The atom type should be changed to 1 and a blank line need to be added after the line "Atoms # full"

######################################  Run the comparison for the tunnel ########################################

centerline_path = f'{path}/{molecule}_centerline_mole.xyz'
centerline_coords = np.genfromtxt(centerline_path, skip_header=2, usecols=(1,2,3))  # shape: (500, 3)
atom_coords, atom_charges = get_charged_atoms(pos_coordinates, neg_coordinates, pho_coordinates) # shape: (n_atoms, 3), shape: (n_atoms,)
atom_charges = atom_charges * 1000
bead_coords = np.array(wall_coordinates)       # shape: (n_beads, 3)
bead_charges = np.array(q_effect) *1000        # shape: (n_beads,)

centerline_distance, atomistic_potentials, cg_potentials = compare_charge_distributions(
    centerline_coords, atom_coords, atom_charges, 
    bead_coords, bead_charges, lamda_d, epsilon_0, epsilon_rel
)

# Create the comparison plot
plot_comparison(centerline_distance, atomistic_potentials, cg_potentials, proportionality,
                save_path=f'charge_distribution_comparison_{proportionality}.png')
