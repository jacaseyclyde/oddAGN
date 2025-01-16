from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import numpy as np
import h5py

from oddagn.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------
    

def load_binary_merger_rate_params(fpath, thin=40):
    with h5py.File(fpath, 'r') as hf:
        bhb_merger_rate_chains = np.array(hf['bhbmf_chains'])

    # thin the parameters
    bhb_merger_rate_chains = bhb_merger_rate_chains[:, ::thin]

    # turn the parameters into a dictionary
    bhb_merger_rate_params = dict(
        gsmf_phi0 = bhb_merger_rate_chains[0],
        gsmf_phi1 = bhb_merger_rate_chains[1],
        gsmf_log10_mbreak = bhb_merger_rate_chains[2],
        gsmf_alpha0 = bhb_merger_rate_chains[3],
        gsmf_alpha1 = bhb_merger_rate_chains[4],
        pair_frac_f0 = bhb_merger_rate_chains[5],
        pair_frac_alpha = bhb_merger_rate_chains[6],
        pair_frac_beta = bhb_merger_rate_chains[7],
        pair_frac_gamma = bhb_merger_rate_chains[8],
        merger_time_tau0 = bhb_merger_rate_chains[9],
        merger_time_alpha = bhb_merger_rate_chains[10],
        merger_time_beta = bhb_merger_rate_chains[11],
        merger_time_gamma = bhb_merger_rate_chains[12],
        mmbulge_beta_mbh = bhb_merger_rate_chains[13],
        mmbulge_alpha_mbh = bhb_merger_rate_chains[14],
        mmbulge_eps_mbh = bhb_merger_rate_chains[15]
    )

    return bhb_merger_rate_params


if __name__ == "__main__":
    app()
