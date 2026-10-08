"""FlowAvatar training and evaluation on AMASS.

Run the entry points as modules from the repository root, for example::

    python -m training.prepare_data --amass_dir /path/to/AMASS --save_dir data/amass_smplh_60fps
    python -m training.train --config training/configs/gru_smplh.yaml
    python -m training.evaluate --config training/configs/gru_smplh.yaml --checkpoint <run>/best_val.pt

See training/README.md for details.
"""
