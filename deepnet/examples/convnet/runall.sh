#!/bin/bash
# Trains DBN on MNIST.
train_deepnet='python ../../trainer.py'
${train_deepnet} model_conv.pbtxt train.pbtxt eval.pbtxt || exit 1

