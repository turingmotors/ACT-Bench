#!/bin/bash

source ../../.venv/bin/activate

# If you haven't logged in yet, you can do so with the following command:
# huggingface-cli login

# Download the model weights
huggingface-cli download turing-motors/Terra video_refiner/refiner_model.pt --local-dir .