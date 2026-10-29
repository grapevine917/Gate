#!/usr/bin/env python3
"""Train state-only 17-128-64-1 MLP with model1's training protocol."""
from train_model1 import main

if __name__ == '__main__':
    main(input_view='state')
