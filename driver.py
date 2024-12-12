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
def alpha():
    produce_alpha_contour("4UG0", 0.05)
    """Alpha command with configurable mode and input file."""

@cli.group()
def mesh():
    """Mesh command group with potential subcommands."""
    pass


@mesh.command("create")
@click.argument('rcsb_id')
def mesh_create(rcsb_id:str):
    """Create a new mesh."""
    # click.echo(f"Creating mesh: {name} with type: {type}")
    create_tunnel_mesh(rcsb_id.upper())


@mesh.command("list")
def mesh_list():
    """List available meshes."""
    click.echo("Listing available meshes...")


@cli.group()
def sim():
    """Simulation command group with potential subcommands."""
    pass


@sim.command("run")
@click.option(
    "--config", type=click.Path(exists=True), help="Configuration file for simulation"
)
@click.option(
    "--output", type=click.Path(), help="Output directory for simulation results"
)
def sim_run(config, output):
    """Run a simulation with optional config and output specifications."""
    click.echo(f"Running simulation")
    if config:
        click.echo(f"Using configuration: {config}")
    if output:
        click.echo(f"Output will be saved to: {output}")


@sim.command("analyze")
@click.argument("result_file", type=click.Path(exists=True))
def sim_analyze(result_file):
    """Analyze simulation results."""
    click.echo(f"Analyzing simulation results from: {result_file}")


if __name__ == "__main__":
    cli()
