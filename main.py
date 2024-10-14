"""
This file provides functions and an example of running the LK scheduling algorithm.
The program generates a random instance and writes it to a file, in the following format:

FILE START
number of machines m
number of jobs n
p1
p2
...
pn
EOF

where pi is the processing time of job i. Afterwards, the file is read into
the program, and used to generate a Schedule object. This is passed into the
Neighborhood object, which is used to optimize the schedule.

Implementation of the LK algorithm and a sanity check implementation of the jump neighborhood
can be found in the file lk_swap.py.
"""
import numpy as np
from lk_swap import *
np.random.seed(0)


def write_random_schedule(n, m):
    fname = "./temp.txt"
    with open(fname, "w") as output:
        output.write(f"{m}\n{n}")
        jobs = np.random.uniform(size=n)
        for j in jobs:
            output.write(f"\n{j}")
    return fname 


def lk_optimize(filename, max_iters=1000):
    sched = Schedule.from_file(filename)
    neighborhood = LinKernighanNeighborhood(sched)
    for it in range(max_iters):
        output = neighborhood.operator()
        if not output:
            break
    return [m.load for m in sched.machines], it + 1


n = 200 # number of jobs
m = 10 # number of machines
filename = write_random_schedule(n, m)

loads_lk, it = lk_optimize(filename)
print(f"LK result: {it} iterations, final makespan = {max(loads_lk)}, final minload = {min(loads_lk)}")


# WIP
#def kswap_optimize(filename, k, max_iters=1000):
#    sched = scheduleStructure.schedule(filename)
#    neighborhood = KSwap.KSwapNeighbourhood(sched, k)
#
#    for it in range(max_iters):
#        output = neighborhood.naiveOperator()
#        if not output or output == "Global Optimal!":
#            break
#    return [m.machineLoad for m in sched.machineList], it + 1

#k = 7
#loads_kswap, it = kswap_optimize(filename, k)
#print(f"{k}-swap result: {it} iterations, final makespan = {max(loads_kswap)}, final minload = {min(loads_kswap)}")
#
#print(f"k-swap - LK = {max(loads_kswap) - max(loads_lk)}")
