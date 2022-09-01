import numpy as np
import matplotlib.pyplot as plt

no_data = np.loadtxt("no_partition.out")
par_data = np.loadtxt("partition.out")
naive_data = np.loadtxt("naive.out")

for bound in [10, 20, 50, 100, 150, 200, 300]:
    array = []
    for item in no_data:
        if item[1] == bound:
            array.append(item)
    array = np.array(array)
    plt.plot(array[:, 0], array[:, 2]/1000000, label=str(bound))

plt.legend(loc="upper left", title="bound")
plt.xlabel("number of triangles")
plt.ylabel("construction time (ms)")
plt.savefig("cons_tris")

plt.clf()
for bound in [100]:
    array = []
    for item in no_data:
        if item[1] == bound:
            array.append(item)
    array = np.array(array)
    plt.plot(array[:, 0], array[:, 5], label="no stream compaction")

for bound in [100]:
    array = []
    for item in par_data:
        if item[1] == bound:
            array.append(item)
    array = np.array(array)
    plt.plot(array[:, 0], array[:, 5], label="stream compaction")

plt.legend(loc="upper left")
plt.xlabel("number of triangles")
plt.ylabel("time of one iteration (ms)")
plt.savefig("par_no")

plt.clf()

for bound in [100]:
    array = []
    for item in naive_data:
        if item[1] == bound:
            array.append(item)
    array = np.array(array)
    plt.plot(array[:, 0], array[:, 4], label="naive traversal")

for bound in [100]:
    array = []
    for item in par_data:
        if item[1] == bound:
            array.append(item)
    array = np.array(array)
    plt.plot(array[:, 0], array[:, 4], label="KD-tree traversal")

plt.legend(loc="upper left")
plt.xlabel("number of triangles")
plt.ylabel("kernel time (ms)")
plt.savefig("naive_kdtree")

plt.clf()
ax1 = plt.gca()
ax2 = ax1.twinx()
bounds = [10, 20, 50, 100, 150, 200, 300]
array = []
for item in par_data:
    if item[0] == 10000:
        array.append(item)

array = np.array(array)
ax1.plot(bounds, array[:, 4], 'b')
ax2.plot(bounds, array[:, 2]/1000000, 'r')
ax1.set_ylabel("kernel time (ms)", color="blue")
ax2.set_ylabel("construction time (ms)", color="red")
ax1.set_xlabel("bound size")
plt.savefig("bound")
