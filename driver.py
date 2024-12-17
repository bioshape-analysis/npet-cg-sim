import click
from alpha_shape.lib import produce_alpha_contour
import dbscan_to_pdb
from mesh_generation.kdtree_approach import create_tunnel_mesh
import sys
sys.dont_write_bytecode = True


@click.group()
def cli():
    """Ribosome nascent peptide exit tunnel (NPET) extraction and simulation cli. See repo documentation and paper for more info."""
    pass


@cli.command()
@click.argument('rcsb_id', type=str)
@click.argument('alpha', type=float)
def alpha(rcsb_id:str, alpha:float=0.05):
    produce_alpha_contour(rcsb_id.upper(), alpha)

@cli.group()
def mesh():
    pass

@mesh.command("create")
@click.argument('rcsb_id')
def mesh_create(rcsb_id:str):
    create_tunnel_mesh(rcsb_id.upper())


@cli.group()
def sim():
    pass

@sim.command("run")
def sim_run():
    click.echo(f"Running simulation")



@cli.group()
def fig():
    pass

@fig.command("one")
def one():
    # slice_tunnel
    # slice_pdb
    # slice_mesh
    # lammps_to_mmcif
    click.echo(f"Producing figure one")

@fig.command("two")
def two():
    # dbscan_to_pdb()
    click.echo(f"Producing figure two")

@fig.command("three")
def three():
    click.echo(f"Producing figure three")


if __name__ == "__main__":
    cli()
