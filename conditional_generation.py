import argparse
import ase.io
from ase import Atoms
from ase.db import connect
import logging
import numpy as np
import os
import pickle
from schnetpack import Properties
import time
import torch
import torch.nn.functional as F

from qm9_data import QM9gen
from utility_functions import (
    cdists,
    get_default_neighbors,
    get_grid,
    update_dict,
)


def get_data(dataset, root_dir, n=1000):
    # get test data
    splits = np.load(os.path.join(root_dir, "split.npz"))
    if dataset == "train":
        qm9_data = QM9gen("data", splits["train_idx"][:n])
    elif dataset == "test":
        qm9_data = QM9gen("data", splits["test_idx"][-n:])
    else:
        raise ValueError("Invalid dataset!")

    # remove one atom per structure
    mol_frags = []
    neighbors = []
    for mol in qm9_data:
        for i in range(mol["_atomic_numbers"].shape[0]):
            if mol['_atomic_numbers'][i] != 1: continue
            new_mol = {}
            # _atomic_numbers
            new_mol["_atomic_numbers"] = torch.cat(
                (mol["_atomic_numbers"][0:i], mol["_atomic_numbers"][i + 1 :])
            )
            # _positions
            new_mol["_positions"] = torch.cat(
                (mol["_positions"][0:i, :], mol["_positions"][i + 1 :, :])
            )
            # _cell
            new_mol["_cell"] = mol["_cell"]
            # _con_mat
            new_mol["_con_mat"] = torch.cat(
                (mol["_con_mat"][0:i, :], mol["_con_mat"][i + 1 :, :])
            )
            new_mol["_con_mat"] = torch.cat(
                (new_mol["_con_mat"][:, 0:i], new_mol["_con_mat"][:, i + 1 :]), dim=1
            )
            curr_neighbors = []
            for j in range(mol["_con_mat"][i].shape[0]):
                if mol["_con_mat"][i, j] > 0:
                    if j < i:
                        curr_neighbors.append(j)
                    else:
                        curr_neighbors.append(j-1)
            neighbors.append(curr_neighbors)
            # _neighbors
            new_mol["_neighbors"] = torch.cat(
                (mol["_neighbors"][:i, :], mol["_neighbors"][i + 1 :, :])
            )
            new_mol["_neighbors"] = torch.cat(
                (new_mol["_neighbors"][:, 0:i], new_mol["_neighbors"][:, i + 1 :] - 1), dim=1
            )
            # _cell_offset
            new_mol["_cell_offset"] = mol["_cell_offset"]
            # _idx
            new_mol["_idx"] = mol["_idx"]

            # add to list of fragments
            mol_frags.append(mol)
    return mol_frags, neighbors


def generate(data, model, device, neighbor_idx=[[]]):
    types = [1, 6, 7, 8, 9]  # retrieve available atom types
    all_types = types + [types[-1] + 1]  # add stop token to list (largest type + 1)

    # generate
    generated = {}
    start_time = time.time()
    with torch.no_grad():
        for frag, neighbors in zip(data, neighbor_idx):
            update_dict(
                generated,
                generate_molecule(
                    model,
                    frag,
                    all_types=all_types,
                    max_length=35,
                    unfinished_idx=neighbors,
                    save_unfinished=True,
                    device=device,
                    max_dist=15.0,
                    n_bins=300,
                    radial_limits=[0.9, 1.7],
                    t=0.1,
                ),  # are we gonna change temperature
            )
        print("")
        end_time = time.time() - start_time
        m, s = divmod(end_time, 60)
        h, m = divmod(m, 60)
        h, m, s = int(h), int(m), int(s)
        print(f"Time consumed: {h:d}:{m:02d}:{s:02d}")

    # sort keys in resulting dictionary
    generated = dict(sorted(generated.items()))

    return generated


def generate_molecule(
    model,
    frag,
    t=0.1,
    max_length=35,
    unfinished_idx=[],
    save_unfinished=True,
    all_types=[1, 6, 7, 8, 9, 10],
    n_bins=300,
    max_dist=15.0,
    radial_limits=[0.9, 1.7],
    device="cuda",
):
    failed_counter = 0
    n_dims = 3
    n_tokens = 2  # token for current atom and for center of mass (start)
    # start_idx = 1  # index of start_token
    n_atoms = frag[Properties.Z].shape[0]
    model = model.to(device)  # put model on chosen device (gpu/cpu)

    # # increase max_length by three to compensate for tokens and last prediction step
    # max_length += n_tokens+1
    all_types = torch.tensor(all_types).long().to(device)
    stop_token = all_types[-1]

    # initialize tensor for atomic numbers
    atom_numbers = torch.zeros(1, max_length)
    atom_numbers[0, 0] = 10
    atom_numbers[0, 1] = 11
    atom_numbers[0, n_tokens:n_tokens+n_atoms] = frag[Properties.Z].expand(1, -1)
    atom_numbers = atom_numbers.long().to(device)
    # initialize tensor for atom positions
    positions = torch.zeros(1, max_length, n_dims)
    positions[0, 1, :] = torch.mean(frag[Properties.R], dim=0).expand(1, 1, -1)
    positions[0, n_tokens:n_tokens+n_atoms, :] = frag[Properties.R].expand(1, -1, -1)
    positions = positions.to(device)

    # initialize tensor that stores the indices of currently focused atom
    current_atoms = torch.ones(1).long().to(device)
    # initialize mask for molecules which are not yet finished (all in the beginning)
    unfinished = torch.ones(1, dtype=torch.bool).to(device)
    # initialize mask to mark single atoms as finished/unfinished
    atoms_unfinished = torch.zeros(1, max_length).float().to(device)
    # mark neighbors as unfinished
    atoms_unfinished[:, unfinished_idx] = 1

    # create grids (a small, linear one for the very first step and a radial one
    # for all following generation steps)
    general_grid, start_grid = get_grid(radial_limits, n_bins=n_bins, max_dist=max_dist)
    general_grid = torch.tensor(general_grid).float().to(device)  # radial grid
    start_grid = torch.tensor(start_grid).float().to(device)  # small start grid

    # create default neighborhood list
    neighbors = torch.tensor(get_default_neighbors(max_length))
    neighbors = neighbors.long().to(device)

    # create dictionary in which generated molecules will be stored (where the key
    # will be the number of atoms in the respective generated molecule)
    results = {}
    # create short name for function that pulls results from gpu and removes
    #  the start and current tokens (first two entries)
    s = lambda x: x[:, n_tokens:].detach().cpu().numpy()

    # define function that builds a model input batch with current state of molecules
    def build_batch(i):
        """i = iteration number"""
        neighbors_i = neighbors[:i, : i-1].expand(1, -1, -1).contiguous()
        neighbor_mask = torch.ones_like(neighbors_i).float()
        # set position of focus token (first entry of positions)
        positions[unfinished, 0] = positions[unfinished, current_atoms[unfinished]]
        # center positions on currently focused atom (for localized grid)
        positions[unfinished, :i] -= positions[unfinished, current_atoms[unfinished]][
            :, None, :
        ]

        # build batch with data of the partial molecules
        batch = {
            Properties.R: positions[unfinished, :i],
            Properties.Z: atom_numbers[unfinished, :i],
            Properties.atom_mask: torch.zeros(1, i, dtype=torch.float),
            Properties.neighbors: neighbors_i,
            Properties.neighbor_mask: neighbor_mask,
            Properties.cell_offset: torch.zeros(1, i, max(i - 1, 1), n_dims),
            Properties.cell: torch.zeros(1, n_dims, n_dims),
            "_next_types": atom_numbers[unfinished, i],
            "_all_types": all_types.view(1, -1),
            "_type_mask": torch.ones(1, i, dtype=torch.float),
        }

        # put batch into torch variables and on gpu
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch

    for i in range(n_tokens+n_atoms, max_length):
        amount = torch.sum(unfinished)
        # stop if the generation process is finished for all molecules
        if amount == 0:
            break
        # store the global state of molecules (whether they are finished)
        global_unfinished = unfinished.clone()

        ### 1st Part ###
        # predict and sample next atom type until all unfinished molecules either have
        # a proper next type (not stop token) or are completely finished
        while torch.sum(unfinished) > 0:
            # set the marker for the current (focus) atom
            current_atoms[unfinished] = torch.multinomial(
                atoms_unfinished[unfinished, :i], 1  # I think there's a problem here; there's nothing stopping the model from picking a stopped atom as focus
            ).squeeze()

            # get batch with updated data (changes in each iteration as unfinished and
            # current_atoms are changed)
            batch = build_batch(i)

            # predict distribution over next atom types with model
            type_pred = F.softmax(model(batch)["type_predictions"], dim=-1)
            # sample types from predictions
            next_types = torch.multinomial(type_pred, 1)
            # store sampled type in tensor with atomic numbers
            atom_numbers[unfinished, i] = all_types[next_types].view(-1)
            # get molecules that predicted no proper type but the stop token
            pred_stop = torch.eq(atom_numbers[unfinished, i], stop_token)
            # set current atom of these molecules to finished
            stop_mask = torch.zeros(len(unfinished), dtype=torch.bool).to(device)
            stop_mask[unfinished] = pred_stop
            atoms_unfinished[stop_mask, current_atoms[stop_mask]] = 0
            # get molecules that were finished in this iteration (those which were
            # unfinished before and now have all atoms marked as finished)
            finished = global_unfinished & (
                torch.sum(atoms_unfinished[:, :i], dim=1) == 0
            )

            # store molecules which are not yet completely finished but have
            # predicted the stop type in the local unfished list in order to repeat
            # the prediction procedure for these molecule (until they predict a
            # proper type for which we can sample a new position)
            unfinished[unfinished] = pred_stop & ~finished[unfinished]

        # store molecules which have been finished in this generation step (i.e. all
        # of their atoms are marked as finished)
        idx = i-n_tokens  # number of atoms in the finished molecules
        if idx > 0 and torch.sum(finished) > 0:
            # center generated molecules on origin token
            positions[finished, :i] -= positions[finished, 1][:, None, :]
            # store positions and atomic numbers in dictionary
            results[idx] = {
                Properties.R: s(positions[finished, :i]),
                Properties.Z: s(atom_numbers[finished, :i]),
                '_idx': frag['_idx']
            }

        # mark finished moleclues in global unfinished mask
        global_unfinished[global_unfinished] = ~finished[global_unfinished]
        # reset local unfinished mask to global state
        unfinished[global_unfinished] = 1

        # stop if max_length of molecules is reached or all are finished
        amount = torch.sum(unfinished)
        if i >= max_length - 1 or amount == 0:
            break

        ### 2nd Part ###
        # sample new position given the type of the next atom

        # get batch with updated data
        batch = build_batch(i)
        # run model to get predictions
        logits = model(batch)
        # get normalized log probabilities
        log_p = F.log_softmax(logits["distance_predictions"], -1)
        del logits

        if i == n_tokens:
            grid = start_grid  # use grid with positions on a line to sample first atom
        else:
            grid = general_grid  # use radial 3d grid for all steps after the first

        # set up storage for log pdf over grid positions
        log_pdf = torch.zeros_like(grid[:, 0].expand(amount, -1))
        step = max_dist / (n_bins - 1)  # step size between two distance bins
        # iterate over atoms in order to reduce memory demands
        for atom in range(i):
            # calculate distances between grid points and respective atom
            dists = cdists(batch[Properties.R][:, atom : atom + 1, :], grid)
            # calculate indices of the corresponding distance bins
            dists += step / 2.0
            dists *= (n_bins - 1) / max_dist
            dists.clamp_(0.0, n_bins - 1)
            dist_labels = dists.long().squeeze(1)
            del dists
            # look up probabilities of distance bins in output
            log_p_grid = torch.gather(log_p[:, atom], -1, dist_labels)

            # multiply predictions for individual atoms to get probability
            log_pdf += log_p_grid
            del log_p_grid
        del log_p

        # normalize distribution over grid
        log_pdf -= torch.logsumexp(log_pdf, -1, keepdim=True)
        # use temperature term on logits and normalize over grid again
        if i > n_tokens:  # not for the very first atom with special grid
            log_pdf /= t
            log_pdf -= torch.logsumexp(log_pdf, -1, keepdim=True)

        log_pdf.exp_()  # take exponential

        # remove numerically failed attempts (NaN in pdf) by marking them as finished
        # (they are not stored among the properly generated molecules, only disregarded)
        if torch.isnan(log_pdf).any():
            failed_mask = torch.isnan(log_pdf).any(dim=-1)
            unfinished[unfinished] = ~failed_mask
            log_pdf = log_pdf[~failed_mask]
            failed_counter += torch.sum(failed_mask)

        # sample positions of new atoms using the calculated pdfs over grid positions
        new_atom_idcs = torch.multinomial(log_pdf, 1).view(-1)
        del log_pdf
        # store new positions
        positions[unfinished, i, :] = grid[new_atom_idcs]

        # # set start token to finished at the end of the first iteration
        # if i == n_tokens:
        #     atoms_unfinished[:, [start_idx]] = 0

    # store unfinished molecules of max_length
    if save_unfinished:
        if torch.sum(unfinished) > 0:
            batch = build_batch(i)
            results[-1] = {
                Properties.R: s(batch[Properties.R]),
                Properties.Z: s(batch[Properties.Z]),
                '_idx': frag['_idx']
            }

    if failed_counter > 0:
        print(f"Failed attempts: {failed_counter}")
    return results

def main(args):
    logging.info("generating molecules...")
    device = torch.device('cuda')
    workdir = "models/gschnet-edm"
    if args.epoch == -1:
        args.epoch = 'best'
    model_path = os.path.join(workdir, "best_model_{}".format(args.epoch))
    os.rename(os.path.join(workdir, "best_model"),
            model_path)
    model = torch.load(model_path).to(device)
    for dataset in ["train", "test"]:
        data, neighbors = get_data(dataset, workdir)
        generated = generate(data, model, device, neighbors)

        gen_path = os.path.join(workdir, "generated/")
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        # get untaken filename and store results
        file_name = os.path.join(gen_path, "generated")
        if os.path.isfile(file_name + ".mol_dict"):
            expand = 0
            while True:
                expand += 1
                new_file_name = file_name + "_" + str(expand)
                if os.path.isfile(new_file_name + ".mol_dict"):
                    continue
                else:
                    file_name = new_file_name
                    break

        for mol_dict in generated.values():
            i = 0
            for nums, positions, idx in zip(mol_dict[Properties.Z], mol_dict[Properties.R], mol_dict['_idx']):
                ase.io.write(f"generated_iclr_cameraready_{dataset}/epoch_{args.epoch}/generated_src{str(int(idx))}_{i}.xyz", Atoms(numbers=nums, positions=positions))
                i += 1

        with open(file_name + ".mol_dict", "wb") as f:
            pickle.dump(generated, f)

    logging.info("...done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch", help="Current epoch", type=int
    )
    args = parser.parse_args()
    main(args)
