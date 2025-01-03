import pandas as pd
import csv
import numpy as np

# Some useful functions

def read_csv_and_extract_coordinates(filename):
    """Reads a CSV file and extracts the last three columns (X, Y, Z) into a list of tuples."""
    # Read the CSV file
    df = pd.read_csv(filename)

    # Extract the last three columns (X, Y, Z)
    coordinates = df[['X', 'Y', 'Z']].to_records(index=False)

    # Convert to a list of tuples
    coords_list = [tuple(coord) for coord in coordinates]

    return coords_list


def place_beads_on_centerline(centerline_coords, bead_distance, n_beads, initiate):
    """Find the beads coordinates of the polypeptide relative to the tunnel centerline."""
    beads = []  # List to store bead positions
    start_point = np.array(centerline_coords[initiate]) # set PTC position
    bead_position = start_point
    beads.append(tuple(bead_position))
    total_beads = 1  # Counter for placed beads
    j = 1

    # Iterate through the centerline coordinates
    while total_beads < n_beads:
        current_distance = 0.0 
        break_loop = 0
        for i in range(j+1,len(centerline_coords)):
            if break_loop == 1:
                break
            end_point = np.array(centerline_coords[i])

            # Calculate the vector and length from start to end point
            direction_vector = end_point - bead_position
            segment_length = np.linalg.norm(direction_vector)
            current_distance = segment_length

            if current_distance == bead_distance:
                bead_position = end_point
                beads.append(tuple(bead_position))
                total_beads += 1
                start_point = bead_position
                j = i
                break_loop = 1
                break

            if current_distance > bead_distance:
                a_point = np.array(centerline_coords[i-1])
                b_point = np.array(centerline_coords[i])
                direction_vector = b_point - a_point
                segment_length = np.linalg.norm(direction_vector)

                # Normalize the direction vector
                if segment_length > 0:
                    for t in np.linspace(0, 1, num=1000):  # Testing 1000 points along AB
                        P = a_point + t * direction_vector  # Calculate point P along AB
                        check_dist = np.linalg.norm(P - start_point) - bead_distance
                        if check_dist > 0 and abs(check_dist) <= 0.005:  
                            bead_position = tuple(P)
                            beads.append(tuple(bead_position))
                            total_beads += 1
                            start_point = bead_position
                            j = i-1
                            break_loop = 1
                            break

            # Check whether the polypeptide has extended out of the tunnel or not
            if current_distance < bead_distance and i == (len(centerline_coords)-1):
                print('warning')
                rest_beads = n_beads - total_beads
                direction_vector = np.array(centerline_coords[-1]) - np.array(centerline_coords[initiate]) 
                segment_length = np.linalg.norm(direction_vector)
                unit_vector = direction_vector / segment_length
                for h in range (rest_beads):
                    P = start_point + (h+1)*bead_distance* unit_vector
                    beads.append(tuple(P))
                    total_beads += 1
                break

    return beads

def write_lammps_data_file(num_atoms, filename, coords):
    """Generate the data file for polypeptide."""
    with open(filename, 'w') as f:

        # Write the header
        f.write("LAMMPS data file. CGCMM style.\n")
        f.write(f" {num_atoms} atoms\n")
        f.write(f" {num_atoms-1} bonds\n")
        f.write(f" {num_atoms-2} angles\n")
        f.write(f" {num_atoms-3} dihedrals\n")
        f.write(" 0 impropers\n")
        f.write(" 2 atom types\n")
        f.write(" 1 bond types\n")
        f.write(" 1 angle types\n")
        f.write(" 1 dihedral types\n")
        f.write(" 0 improper types\n")
        f.write("\n")
        f.write(" 0  1000  xlo xhi          #-2.950000 9.868000\n")
        f.write(" 0  1000  ylo yhi         # -6.784000 11.824000\n")
        f.write(" 0  1000  zlo zhi         #-11.057000 7.885000\n")
        f.write("\n")
        
        # Write Masses
        f.write(" Masses\n\n")
        f.write(" 1 12.000 # CH2\n")
        f.write(" 2 12.000\n")
        f.write("\n")
        
        # Write Atoms
        f.write(" Atoms # full\n\n")
        
        for i in range(1, num_atoms + 1):
            # Example atom data; modify as needed
            j  = 2 if (i - 2) % 3 == 0 else 1
            f.write(f"{i} 1 {j} 0.000000 {round(coords[i-1][0], 4)} {round(coords[i-1][1], 4)} {round(coords[i-1][2], 4)} # C UNL\n")
        
        # Write Bonds
        f.write("\n Bonds\n\n")
        for i in range(1, num_atoms):
            f.write(f"{i} 1 {i} {i + 1}\n")
        
        # Write Angles
        f.write("\n Angles\n\n")
        for i in range(1, num_atoms-1):
            f.write(f"{i} 1 {i} {i + 1} {i + 2}\n")
        
        # Write Dihedrals
        f.write("\n Dihedrals\n\n")
        for i in range(1, num_atoms-2):
            f.write(f"{i} 1 {i} {i + 1} {i + 2} {i + 3}\n")


# Example usage

molecule = "4UG0"

# Read centerline coordinates for each tunnel from MOLE2.5
centerline_coords = read_csv_and_extract_coordinates(f'/Users/Desktop/MOLE_{molecule}_tunnel.csv' )

# Number of beads in the polypeptide (residue number = num_beads/3)
num_beads = 40*3     #40 amino acid residues

# Bond length in angstroms
bead_distance = 1.529  

bead_coordinates = place_beads_on_centerline(centerline_coords, bead_distance, num_beads, 5)

filename = f'/Users/Desktop/data.lj_{molecule}' 
write_lammps_data_file(num_beads, filename, bead_coordinates)

print(f"{filename} has been created.")
