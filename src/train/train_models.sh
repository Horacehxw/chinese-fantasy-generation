#!/usr/bin/env bash
python train_lm.py
python train_flat_vae.py
python train_flat_vae.py --kl_annealing cyclic

