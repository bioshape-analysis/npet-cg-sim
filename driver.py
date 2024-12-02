import click
from mesh_generation.kdtree_approach import main
import sys
sys.dont_write_bytecode = True

@click.group()
def cli():
    """Ribosome nascent peptide exit tunnel (NPET) extraction and simulation cli. See repo documentation and paper for more info."""
    pass

@cli.command()
@click.option('--mode', default='default', help='Operation mode for alpha command')
@click.argument('input_file', type=click.Path(exists=True))
def alpha(mode, input_file):
    """Alpha command with configurable mode and input file."""
    click.echo(f"Alpha command executed with mode: {mode}")
    click.echo(f"Input file: {input_file}")

@cli.group()
def mesh():
    """Mesh command group with potential subcommands."""
    pass

@mesh.command('create')
# @click.option('--name', required=True, help='Name of the mesh to create')
# @click.option('--type', default='default', help='Type of mesh')
def mesh_create():
    """Create a new mesh."""
    # click.echo(f"Creating mesh: {name} with type: {type}")
    main()
    

@mesh.command('list')
def mesh_list():
    """List available meshes."""
    click.echo("Listing available meshes...")

@cli.group()
def sim():
    """Simulation command group with potential subcommands."""
    pass

@sim.command('run')
@click.option('--config', type=click.Path(exists=True), help='Configuration file for simulation')
@click.option('--output', type=click.Path(), help='Output directory for simulation results')
def sim_run(config, output):
    """Run a simulation with optional config and output specifications."""
    click.echo(f"Running simulation")
    if config:
        click.echo(f"Using configuration: {config}")
    if output:
        click.echo(f"Output will be saved to: {output}")

@sim.command('analyze')
@click.argument('result_file', type=click.Path(exists=True))
def sim_analyze(result_file):
    """Analyze simulation results."""
    click.echo(f"Analyzing simulation results from: {result_file}")

if __name__ == '__main__':
    cli()