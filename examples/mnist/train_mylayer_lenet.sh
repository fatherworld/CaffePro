#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/mylayer_lenet_solver.prototxt $@
