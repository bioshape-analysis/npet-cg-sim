import os

class StructureAssets():

    rcsb_id   : str
    data_dir  : str
    structpath: str


    def __init__(self, data_dir:str, rcsb_id:str) -> None:
        self.rcsb_id = rcsb_id.upper()
        self.data_dir = data_dir
        assert os.path.exists(data_dir)
        self.structpath = os.path.join(data_dir, rcsb_id.upper())
        pass


    @property
    def cif_struct(self):
        return os.path.join(self.structpath, '{}.cif'.format(self.rcsb_id))

    @property
    def tunnel_pcd_normal_estimated(self):
        return os.path.join(self.structpath, '{}_tunnel_pcd_normal_estimated.ply'.format(self.rcsb_id))

    @property
    def ashape_watertight(self):
        return os.path.join(self.structpath, '{}_ashape_watertight.ply'.format(self.rcsb_id))

    @property
    def tunnel_mesh(self):
        return os.path.join(self.structpath, '{}_tunnel_poisson_recon.ply'.format(self.rcsb_id))



