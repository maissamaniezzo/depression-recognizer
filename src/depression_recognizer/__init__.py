"""Depression recognition from resting-state fMRI using Med3D-ResNet18.

Pipeline:
    1. preprocessing.bids_to_3d  — fMRI 4D -> (X,Y,Z,3) NIfTI with [ALFF, fALFF, ReHo]
    2. data.split                — stratified train/validation split + manifest.csv
    3. modeling.train            — fine-tune Med3D ResNet18 (1->3 channels, custom head)
    4. modeling.infer            — evaluate / predict on a split
"""

__version__ = "0.1.0"
