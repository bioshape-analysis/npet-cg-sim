import re

# Parameters
epsilon_0 = 8.854*10**-12 #F/m
epsilon_rel = 40  #average value of protein, rna, and water 
lamda_d = 3

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



# 1. Example usage (for tunnel wall particles only)

molecule = '4UG0'

# Store the coordinates of positively charged resdiues within 50 angstrom from the tunnel centerline derived from PyMOL
pos_coordinates = []

# Read and process the file
with open(f'/Users/Desktop/50_pos_{molecule}.txt', 'r') as file:
    coordinates = []
    for line in file:
        cleaned_line = re.sub(r"\\", "", line).strip()
        coords = tuple(map(float, cleaned_line.split()))
        coordinates.append(coords)

# Remove duplicate coordinates
pos_coordinates = list(map(list, set(coordinates)))

# Store the coordinates of negatively charged resdiues within 50 angstrom from the tunnel centerline derived from PyMOL
neg_coordinates = []

# Read and process the file
with open(f'/UsersDesktop/50_neg_{molecule}.txt', 'r') as file:
    coordinates = []
    for line in file:
        cleaned_line = re.sub(r"\\", "", line).strip()
        coords = tuple(map(float, cleaned_line.split()))
        coordinates.append(coords)

# Remove duplicate coordinates
neg_coordinates = list(map(list, set(coordinates)))
print(len(neg_coordinates),neg_coordinates[0])

# Store the coordinates of negatively charged phosphorus atoms within 50 angstrom from the tunnel centerline derived from PyMOL
pho_coordinates = []

# Read and process the file
with open(f'/Users/Desktop/50_phos_{molecule}.txt', 'r') as file:
    coordinates = []
    for line in file:
        cleaned_line = re.sub(r"\\", "", line).strip()
        coords = tuple(map(float, cleaned_line.split()))
        coordinates.append(coords)

# Remove duplicate coordinates
pho_coordinates = list(map(list, set(coordinates)))
print(len(pho_coordinates),pho_coordinates[0])

# Read the tunnel wall particle coordinates (new data file generated by LAMMPS after removing overlapping particles)
wall_coordinates = []
atom_ids = []

# Define flags to track the target section
in_target_section = False

# Read the file line by line
with open(f"/Users/Desktop/data.{molecule}_tunnel_new", "r") as file:
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
    q_e = V_wall[i]*8*np.pi*epsilon_0*5*3.5*10**-10/(3*(1.6*10**-19)) # Assunmption for tunnel wall particles only
    q_effect.append(q_e)


# Generate a new data file for tunnel wall particles with effective charges
atom_charge_map = dict(zip(atom_ids, q_effect))

output_lines = []

# Flags to track the section of the file
in_target_section = False

# Open the file to read and the output file to write
with open(f"/Users/Desktop/data.{molecule}_tunnel_new", "r") as infile, open(f"/Users/Desktop/data.{molecule}_tunnel_charged", "w") as outfile:
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

print("File has been processed and saved as 'output_file.txt', need futher manipulate modify")
# Be aware that the new data file need to be mannually modifed a bit. 
# The atom type should be changed to 1 and a blank line need to be added after the line "Atoms # full"


# 2. Example usage (for ribosome surface wall particles only)

# Read the ribsome surface particle coordinates (new data file generated by LAMMPS after removing overlapping particles)
sphere_coordinates = []
atom_ids = []

# Define flags to track the target section
in_target_section = False

# Read the file line by line
with open("/Users/Desktop/data.{molecule}_sphere_new", "r") as file:
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
grid_positions, data_array, matched_data = parse_opendx(f"/Users/Desktop/{molecule}_map.dx")

# Calculate the interpolated potentials at the ribosome surface wall particle coordinates
# Copy those values from map.dx file
grid_origin = [21.78401,  -0.527997,  9.276]
grid_deltas = [[3.163917, 0, 0], [0, 3.340531, 0], [0, 0, 3.165094]]
grid_counts = (97, 97, 97)
data = data_array  
query_points = sphere_coordinates

interpolated_values = interpolate_spline(grid_origin, grid_deltas, grid_counts, data, query_points)*0.0256 # Change the units from kT/e to V

# Assign the effective charges to the ribosome surface wall particles according to interpolated potentials
V_sphere = interpolated_values
q_sphere = []
for i in range (len(V_sphere)):
    q_e = V_sphere[i]*8*np.pi*epsilon_0*0.5*3.5*10**-10/(3*(1.6*10**-19)) # Assunmption for ribosome surface wall particles only
    q_sphere.append(q_e)


# Generate a new data file for ribosome surface wall particles with effective charges
atom_charge_map = dict(zip(atom_ids, q_sphere))

output_lines = []

# Flags to track the section of the file
in_target_section = False

# Open the file to read and the output file to write
with open(f"/Users/Desktop/data.{molecule}_sphere_new", "r") as infile, open(f"/Users/Desktop/data.{molecule}_sphere_charged", "w") as outfile:
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

print("File has been processed and saved as 'output_file.txt', need futher manipulate modify")
# Be aware that the new data file need to be mannually modifed a bit. 
# The atom type should be changed to 1 and a blank line need to be added after the line "Atoms # full"
