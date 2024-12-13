import os
import numpy as np
from Bio import PDB
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Structure import Structure
import json
from typing import Dict, List, Tuple, Set
from pathlib import Path

from data.asset_manager import StructureAssets
from mesh_generation.util import landmark_constriction_site, landmark_ptc

def load_residue_indices(json_path: str) -> Dict[str, List[int]]:
    """Load residue indices from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def get_residue_coordinates(structure: Structure, 
                          chain_residues: Dict[str, List[int]]) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Get coordinates of CA atoms for specified residues in each chain.
    
    Returns:
        Dict mapping chain_id -> {residue_id -> CA_coordinates}
    """
    coords_dict = {}
    
    for chain in structure[0]:
        chain_id = chain.id
        if chain_id not in chain_residues:
            continue
            
        coords_dict[chain_id] = {}
        target_residues = set(chain_residues[chain_id])
        
        for residue in chain:
            if residue.id[1] in target_residues:  # residue.id[1] is the residue number
                try:
                    # ca_atom = residue.center_of_mass()
                    # coords_dict[chain_id][residue.id[1]] = ca_atom.get_coord()
                    coords_dict[chain_id][residue.id[1]] = residue.center_of_mass()
                except KeyError:
                    print(f"Warning: No CA atom found for residue {residue.id[1]} in chain {chain_id}")
                    
    return coords_dict

def filter_residues_by_plane(coords_dict: Dict[str, Dict[int, np.ndarray]],
                           base_point: np.ndarray,
                           axis_point: np.ndarray,
                           angle: float) -> Dict[str, List[int]]:
    """
    Filter residues based on their position relative to the slicing plane.
    
    Returns:
        Dict mapping chain_id -> list of residue numbers that fall on the kept side
    """
    # Calculate plane normal
    axis = axis_point - base_point
    axis_unit = axis / np.linalg.norm(axis)
    
    if not np.allclose(axis_unit, [1, 0, 0]):
        init_perp = np.cross(axis_unit, [1, 0, 0])
    else:
        init_perp = np.cross(axis_unit, [0, 1, 0])
    init_perp = init_perp / np.linalg.norm(init_perp)
    
    # Rotate the initial perpendicular vector around the axis
    rotation_matrix = rotation_matrix_from_axis_angle(axis_unit, np.radians(angle))
    plane_normal = rotation_matrix @ init_perp
    
    filtered_residues = {}
    
    for chain_id, residue_coords in coords_dict.items():
        kept_residues = []
        
        for res_id, coords in residue_coords.items():
            # Calculate signed distance from point to plane
            point_centered = coords - base_point
            distance = np.dot(point_centered, plane_normal)
            
            # Keep residues on the negative side of the plane
            if distance <= 0:
                kept_residues.append(res_id)
        
        if kept_residues:
            filtered_residues[chain_id] = sorted(kept_residues)
    
    return filtered_residues

def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create a rotation matrix for rotating around an axis by an angle."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    R = (np.eye(3) + np.sin(angle) * K + 
         (1 - np.cos(angle)) * (K @ K))
    
    return R

def slice_mmcif_structure(mmcif_path: str,
                         residue_indices_path: str,
                         base_point: np.ndarray,
                         axis_point: np.ndarray,
                         angle: float = 45) -> Dict[str, List[int]]:
    """
    Main function to slice mmCIF structure and filter residues.
    
    Parameters:
        mmcif_path: Path to mmCIF file
        residue_indices_path: Path to JSON file with residue indices
        base_point: Starting point of cylinder axis
        axis_point: End point of cylinder axis
        angle: Angle in degrees for the slicing plane rotation
    
    Returns:
        Dictionary mapping chain IDs to lists of kept residue numbers
    """
    # Load structure and residue indices
    parser = MMCIFParser()
    structure = parser.get_structure('ribosome', mmcif_path)
    chain_residues = load_residue_indices(residue_indices_path)
    
    # Get coordinates of specified residues
    coords_dict = get_residue_coordinates(structure, chain_residues)
    
    # Filter residues based on slicing plane
    filtered_residues = filter_residues_by_plane(coords_dict, base_point, axis_point, angle)
    
    return filtered_residues

if __name__ == "__main__":
    RCSB_ID = '4UG0'
    data_dir = os.getenv('DATA_DIR')
    # Example usage
    mmcif_path           = StructureAssets(data_dir, RCSB_ID).cif_struct
    residue_indices_path = "cylinder_residues.json"
    
    # Define cylinder axis points (replace with your actual points)
    base_point          = np.array(landmark_ptc(RCSB_ID))
    axis_point = np.array(landmark_constriction_site(RCSB_ID))
    
    # Slice structure and get filtered residues
    filtered_residues = slice_mmcif_structure(
        mmcif_path,
        residue_indices_path,
        base_point,
        axis_point
    )
    
    # Save results
    output_path = "half_cylinder_residues.json"
    with open(output_path, 'w') as f:
        json.dump(filtered_residues, f, indent=4)