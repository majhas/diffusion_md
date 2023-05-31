import numpy as np
import mdtraj as mdt

def read_trajectories(traj_filepath, pdb_filepath):
    data = np.load(traj_filepath)

    # pdb_structure = mdt.load(pdb_filepath).topology

    # trajectories = mdt.Trajectory(data['coords'], pdb_structure)
    return data['coords']


def compute_phi(trajectories):
    return mdt.compute_phi(traj=trajectories)


def compute_psi(trajectories):
    return mdt.compute_psi(traj=trajectories)


def compute_phi_and_psi(trajectories):
    return compute_phi(trajectories), compute_psi(trajectories)
