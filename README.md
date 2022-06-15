# MOMP
 Multidimensional matching pursuit algorithm

# How to install
You can install the latest version through pip
```
pip install git+https://github.com/WiSeCom-Lab/MOMP-core
```

# How to use
You will first have to define the algorithm through three pieces: the algorithm structure, the projection step and the stop condition
```
algorithm = core(projection, stop)
```
Once the algorithm is defined you can simply pass the data you want to decompose to get the sparse decomposition index and values
```
I, alpha = algorithm(data)
```

# Algorithm core
Found in `MOMP.mp`, the core indicating the workflow of the algorithm.
It can either be `MP` for plain matching pursuit or `OMP` for orthogonal matching pursuit.

# Projection step
Found in `MOMP.proj`, the projection step is the main innovation in MOMP, this one can be
## OMP_proj
`OMP_proj(A, X)` is the classic OMP projection step for the measurement matrix A and the dictionary X
## MOMP_proj
`MOMP_proj(A, X)` is the MOMP projection step for the measurement matrix A and the dictionaries collection X

# Stop criteria
Found in `MOMP.stop`, the stop criteria determines when to stop the algorithm run.
## General
`General(maxIter)` determines the maximum number of algorithm iterations
