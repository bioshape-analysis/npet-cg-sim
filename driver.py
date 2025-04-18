import click
from alpha_shape.lib import produce_alpha_contour
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


if __name__ == "__main__":
    cli()
