# mcmc_tutorial

This is a short tutorial for using emcee. The requirements are a working
installation of `mpich`, `mpi4py` and `emcee`. The first package can be
obtained from the package manager on your laptop and it should exist on your
cluster machines. The others can be installed using pip. First load the correct
python module and then run pip installs.
```
module load anaconda3
pip install --user mpi4py
pip install --user emcee
```

On a cluster such as pegasus, you have to run
```
module load gcc
module load mpich
module load anaconda3

python3 single_processor_code.py
mpirun -np 32 python3 mpicode.py
```
