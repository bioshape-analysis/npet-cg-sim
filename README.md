# Nascent Peptide Exit Tunnel (NPET) Coarse Grained simulation protocol

This is a satelite repo for the paper [Advanced coarse-grained model of the ribosome exit tunnel for fast simulation of the nascent polypeptide chain dynamics](#biophys_url_to_be_added).

## Installation


Note: `python3.11` is the latest version that is supported (this is solely due to `open3d` dependency) .


```
pip install -r requirements.txt
```

Compile the Poisson Surface reconstruction binary from here: https://github.com/mkazhdan/PoissonRecon 


## Environment

- `POISSON_RECON_BIN`: the _fully-qualified_ path to it to the environment as `POISSON_RECON_BIN` variable.
- `DATA_DIR`: directory containing initial `.mmcif` strucutres and intermediate artifacts of the pipeline.

You may copy `.env.example` to `.env` and modify the variables to suit your setup.


## Usage

The main `driver.py` is the entry cli to the mesh generation utilities:
```
# to produce interior tunnel mesh of the 4UG0 ribosome structure
python driver.py mesh mesh_create 4UG0

# to produce an alphashape-like contour of the 4UG0 ribosome structure
python driver.py alpha  4UG0
```

To place the CG beads on the mesh structures of the tunnel and ribosome, follow the steps in 
```
# under the "CG_model" folder
`wall_data_file.py`
```

To map effective charges onto the CG beads of the tunnel and ribosome, follow the steps in 
```
# under the "CG_model" folder
`charged_wall.py`
```

Two in.files in the "CG_model/LAMMPS/refinement" folder should be read sequentially by LAMMPS to reduce the size of the simulation system by removing overlapping CG beads:
```
lmp_serial < in.merge_sphere
lmp_serial < in.merge_tunnel
```



