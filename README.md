This repository contains the implementation of an equivariant neural network architecture for molecular and atomic
multipole prediction. Large parts are taken from the [mlff](https://github.com/thorben-frank/mlff/) repository,
in particular the network layer implementation of the So3krates network. My
personal contributions are most of the observable modules in /src/nn/observable/observable.py and other small helper
functions. The class CeqNet is an adaptation of the StackNet class from mlff to the multipole prediction task.
Example trainings implementations will follow soon.