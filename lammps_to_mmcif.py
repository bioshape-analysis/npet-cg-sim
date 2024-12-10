def read_lammps_file(filename):
    """Read LAMMPS trajectory file and extract atom coordinates."""
    atoms = []
    box_bounds = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "ITEM: BOX BOUNDS pp pp pp":
            # Read box bounds
            for _ in range(3):
                i += 1
                xlo, xhi = map(float, lines[i].strip().split())
                box_bounds.append((xlo, xhi))
        elif line == "ITEM: ATOMS id type xs ys zs":
            # Read atoms
            i += 1
            while i < len(lines) and lines[i].strip():
                parts = lines[i].strip().split()
                if len(parts) >= 5:
                    atom_id = int(parts[0])
                    atom_type = int(parts[1])
                    x, y, z = map(float, parts[2:5])
                    atoms.append((atom_id, atom_type, x, y, z))
                i += 1
        i += 1
    
    return atoms, box_bounds

def write_pdb(atoms, box_bounds, output_file):
    """Write atoms to PDB format."""
    with open(output_file, 'w') as f:
        # Write CRYST1 record for unit cell
        a, b, c = [bounds[1] - bounds[0] for bounds in box_bounds]
        f.write(f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}  90.00  90.00  90.00 P 1           1\n")
        
        # Write atoms
        for i, (atom_id, atom_type, xs, ys, zs) in enumerate(atoms, 1):
            # Convert scaled coordinates to absolute coordinates
            x = xs * (box_bounds[0][1] - box_bounds[0][0])
            y = ys * (box_bounds[1][1] - box_bounds[1][0])
            z = zs * (box_bounds[2][1] - box_bounds[2][0])
            
            # Use HETATM for non-standard residues
            # Using element 'C' for simplicity, you can modify based on atom_type if needed
            f.write(f"HETATM{i:5d}  C   MOL {i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          C\n")
        
        f.write("END\n")

def convert_lammps_to_pdb(input_file, output_file):
    """Convert LAMMPS trajectory file to PDB format."""
    atoms, box_bounds = read_lammps_file(input_file)
    write_pdb(atoms, box_bounds, output_file)

if __name__ == "__main__":
    input_file = "tunnel.lammpstrj"
    output_file = "output_tunnel.pdb"           # Replace with desired output file name
    convert_lammps_to_pdb(input_file, output_file)