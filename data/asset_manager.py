import os


class StructureAssets:

    rcsb_id: str
    data_dir: str
    structpath: str

    def __init__(self, data_dir: str, rcsb_id: str) -> None:
        self.rcsb_id = rcsb_id.upper()
        self.data_dir = data_dir
        assert os.path.exists(data_dir)
        self.structpath = os.path.join(data_dir, rcsb_id.upper())
        pass

    @property
    def cif_struct(self):
        return os.path.join(self.structpath, "{}.cif".format(self.rcsb_id))

    @property
    def tunnel_pcd_normal_estimated(self):
        return os.path.join(
            self.structpath, "{}_tunnel_pcd_normal_estimated.ply".format(self.rcsb_id)
        )

    # Alpha surface and tunnel mesh and their clipped versions
    @property
    def ashape_watertight(self):
        return os.path.join(
            self.structpath, "{}_ashape_watertight.ply".format(self.rcsb_id)
        )

    @property
    def tunnel_mesh(self):
        return os.path.join(
            self.structpath, "{}_tunnel_poisson_recon.ply".format(self.rcsb_id)
        )

    @property
    def tunnel_half_mesh(self):
        return os.path.join(
            self.structpath, "{}_tunnel_half_poisson_recon.ply".format(self.rcsb_id)
        )

    @property
    def ashape_half_mesh(self):
        return os.path.join(
            self.structpath, "{}_half_ashape_watertight.ply".format(self.rcsb_id)
        )

    # These are just to visualize lamps trajectories/pointclouds as pdb files.

    @property
    def lammps_traj_tunnel(self):
        return os.path.join( self.structpath, "{}_tunnel.lammpstraj".format(self.rcsb_id) )

    @property
    def lammps_traj_ashape(self):
        return os.path.join( self.structpath, "{}_ashape.lammpstraj".format(self.rcsb_id) )

    @property
    def lammps_traj_tunnel_as_pdb(self):
        return os.path.join( self.structpath, "{}_tunnel.lammpstraj.pdb".format(self.rcsb_id) )

    @property
    def lammps_traj_ashape_as_pdb(self):
        return os.path.join( self.structpath, "{}_ashape.lammpstraj.pdb".format(self.rcsb_id) )

    @property
    def dbscan_clusters_PDB_noise(self):
        return os.path.join( self.structpath, "{}_dbscan_clusters.noise.pdb".format(self.rcsb_id) )

    @property
    def dbscan_clusters_PDB_refined(self):
        return os.path.join( self.structpath, "{}_dbscan_clusters.refined.pdb".format(self.rcsb_id) )
    @property
    def dbscan_clusters_PDB_largest(self):
        return os.path.join( self.structpath, "{}_dbscan_clusters.largest.pdb".format(self.rcsb_id) )

    @property
    def dbscan_clusters_mmcif_noise(self):
        return os.path.join( self.structpath, "{}_dbscan_clusters.noise.cif".format(self.rcsb_id) )

    @property
    def dbscan_clusters_mmcif_refined(self):
        return os.path.join( self.structpath, "{}_dbscan_clusters.refined.cif".format(self.rcsb_id) )
    @property
    def dbscan_clusters_mmcif_largest(self):
        return os.path.join( self.structpath, "{}_dbscan_clusters.largest.cif".format(self.rcsb_id) )

    @property
    def dbscan_clusters_xyz_noise(self):
        return os.path.join( self.structpath, "{}_dbscan_clusters.noise.xyz".format(self.rcsb_id) )

    @property
    def dbscan_clusters_xyz_refined(self):
        return os.path.join( self.structpath, "{}_dbscan_clusters.refined.xyz".format(self.rcsb_id) )
    @property
    def dbscan_clusters_xyz_largest(self):
        return os.path.join( self.structpath, "{}_dbscan_clusters.largest.xyz".format(self.rcsb_id) )