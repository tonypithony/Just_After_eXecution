# JAX ("NumPy on steroids") - Just After eXecution

# pip install --upgrade "jax[cpu]"

# pip install --upgrade "jax[cuda12]"

import numpy as np
from jax import jit
import jax.numpy as jnp
from time import time


# define the cube function
def cube(x):
    return x**100_000_000

# generate data
# x = jnp.ones((100_000, 100_000))
x = np.random.rand(10_000, 10_000)
# x = jnp.array(x)

# create the jit version of the cube function
jit_cube = jit(cube)

# apply the cube and jit_cube functions to the same data for speed comparison
start = time()
print(cube(x), time()-start)

x = jnp.array(x)
start = time()
print(cube(x), time()-start)

start = time()
print(jit_cube(x), time()-start)
'''
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]] 5.484015703201294
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]] 0.0445096492767334
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]] 0.04341006278991699
 '''