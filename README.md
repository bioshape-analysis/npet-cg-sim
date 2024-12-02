# Nascent Peptide Exit Tunnel (NPET) Coarse Grained simulation protocol

This is a satelite repo for the paper [Advanced coarse-grained model of the ribosome exit tunnel for fast simulation of the nascent polypeptide chain dynamics](#citation_url).

## Installation


Note: `python3.11` is the latest version that is supported (this is solely due to `open3d` dep.) 
pip install -r reqs.txt
Compile the Poisson Surface reconstruction binary from here https://github.com/mkazhdan/PoissonRecon 


## Environment

- `POISSON_RECON_BIN`: the _fully-qualified_ path to it to the environment as `POISSON_RECON_BIN` variable.
- `DATA_DIR`: directory containing initial `.mmcif` strucutres and intermediate artifacts of the pipeline.

You may copy `.env.example` to `.env` and modify the variables to suit your setup.


`DATA_DIR` layout is as follows:
```
```

----------------------------------
### Layout prototype during development:

```
.
├── README.md
├── lammps_workflow
│   └── Shiqi's LAMMPS workflow
├── alpha_shape 
│   └── Wenjun's alpha shape code
├── mesh_generation
│   └── Artem mesh gen code
├── data             
├── examples
├── cli.py 
└──requirements.txt
```