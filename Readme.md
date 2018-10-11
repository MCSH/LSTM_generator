# LSTM Text Generator

This code implements a configurable LSTM to generate text similiar to a set you train it on.

## Requirements

You need any version of python with `numpy` and `keras` installed. Note that for `keras` to work you need to either install `tensorflow` or `theano` or `CTNK`.

Configure the `config.py` file to suit your needs and provide a dataset under the name `input.txt` (changable in the configs).

If there is a model file, the neural network will continue working on that, otherwise it will start from scratch. In order to change the structure of the machine you need to clear out the old model file.

Run `textgen.py` to train the model and run `run.py` in order to print out samples based on the trained model.

## TODO
* Save the iteration for the model, so that when resumed, it would continue counting and not reset to 1.
* Save the model name based on the input file
* Write a better Readme
* Write better normalizer
* Provide some sample datasets
* Write more TODOs
