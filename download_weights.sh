#!/usr/bin/env bash

mkdir -p model_weights

# Original DeDoDe weights
wget -P model_weights/ https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth
wget -P model_weights/ https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth

# Our weights
wget -P model_weights/ https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_C4_steerer_setting_A.pth

exit 0
