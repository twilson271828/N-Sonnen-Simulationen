#!/usr/bin/bash

g++ armaspeed.cpp -o armaspeed -O3 -march=native -fopenmp -larmadillo
