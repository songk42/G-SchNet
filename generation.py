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
from utility_functions import update_dict, generate_molecules, get_dict_count


def generate(args, model, device):
    # generate molecules (in chunks) and print progress

    dataclass = QM9gen
    types = sorted(dataclass.available_atom_types)  # retrieve available atom types
    all_types = types + [types[-1] + 1]  # add stop token to list (largest type + 1)
    start_token = types[-1] + 2  # define start token (largest type + 2)
    amount = args.num_mols
    chunk_size = args.chunk_size
    if chunk_size >= amount:
        chunk_size = amount

    # set parameters for printing progress
    if int(amount / 10.0) < chunk_size:
        step = chunk_size
    else:
        step = int(amount / 10.0)
    increase = lambda x, y: y + step if x >= y else y
    thresh = step
    progress = lambda x, y: print(
        f"\x1b[2K\rSuccessfully generated" f" {x}", end="", flush=True
    )

    # generate
    generated = {}
    left = args.num_mols
    done = 0
    start_time = time.time()
    with torch.no_grad():
        while left > 0:
            if left - chunk_size < 0:
                batch = left
            else:
                batch = chunk_size
            generated_results, unfinished = generate_molecules(
                batch,
                model,
                all_types=all_types,
                start_token=start_token,
                max_length=35,
                save_unfinished=args.save_unfinished,
                device=device,
                max_dist=15.0,
                n_bins=300,
                radial_limits=dataclass.radial_limits,
                t=0.1,
            )
            update_dict(
                generated,
                generated_results,
            )
            left -= (batch - unfinished)
            done += (batch - unfinished)
            n = np.sum(get_dict_count(generated, 35))
            progress(n, thresh)
            thresh = increase(n, thresh)
        print("")
        end_time = time.time() - start_time
        m, s = divmod(end_time, 60)
        h, m = divmod(m, 60)
        h, m, s = int(h), int(m), int(s)
        print(f"Time consumed: {h:d}:{m:02d}:{s:02d}")

    # sort keys in resulting dictionary
    generated = dict(sorted(generated.items()))

    return generated

def main(args):
    logging.info("generating molecules...")
    device = torch.device('cuda')
    workdir = "models/gschnet-100k-10k"  # TODO CHANGED
    if args.epoch == -1:
        args.epoch = 'best'
    model_path = os.path.join(workdir, "best_model_{}".format(args.epoch))
    os.rename(os.path.join(workdir, "best_model"),
            model_path)
    model = torch.load(model_path).to(device)
    generated = generate(args, model, device)

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

    ase_dir = f"generated_100k_10k/epoch_{args.epoch}"  # TODO CHANGED
    if not os.path.exists(ase_dir):
        os.makedirs(ase_dir)

    for mol_dict in generated.values():
        for idx, (nums, positions) in enumerate(zip(mol_dict[Properties.Z], mol_dict[Properties.R])):
            ase.io.write(os.path.join(ase_dir, f"generated_src{idx}.xyz"), Atoms(numbers=nums, positions=positions))

    with open(file_name + ".mol_dict", "wb") as f:
        pickle.dump(generated, f)

    logging.info("...done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch", help="Current epoch", type=int
    )
    parser.add_argument(
        "--num_mols", help="Number of molecules to generate", type=int
    )
    parser.add_argument(
        "--chunk_size", help="Chunk size", type=int
    )
    parser.add_argument(
        "--save_unfinished", help="Save unfinished fragments", type=bool, default=False
    )
    args = parser.parse_args()
    main(args)
