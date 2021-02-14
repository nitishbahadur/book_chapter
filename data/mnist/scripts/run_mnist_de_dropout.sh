#!/bin/sh
sbatch -o ../data/mnist/logs/mnist_de_dropout_5e-5-64_2_2.log mnist_de_dropout.py 5e-5 64 31 64 0.2 0.2