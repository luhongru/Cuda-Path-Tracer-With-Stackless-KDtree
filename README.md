# Cuda Path Tracer With Stackless KDtree
This repository implements a path tracer with tracer and a CPU contructed, GPU traversed KD-tree acceleration structure. The path tracer uses cuda kernels to shade the images and openGL to render.

## Stackless KD-Tree
KD-Tree is an efficient acceleration structure for static scenes in ray tracing, and GPUs provide great performance in large throughput parallel computation. However, an intuitive contruction of KD-Tree requires a recursive traversal (or stack-based) which is poorly supported on GPUs. In order to exploit the computation power of GPUs, multiple designs of stackless KD-tree are proposed. This project implements a stackless Kd-tree presented by Stefan et al. which augment the tree nodes with "ropes". A tree node has one rope for each of face of its bounding box. And the rope connects to a neighboring node on that face and the corresponding face of the neighboring node must be larger. A figure from the paper by Stefan provides a good illustration here.
![alt text](./img/ropes.PNG?raw=true)
Looking at the right face of node 2 (with blue rope), it is adjacent to both node 4 and node 6. Therefore, this blue rope must point to a common ancestor node of node 4 and node 6, which is the blue node in the figure.

The goal of traversing a KD-tree is to find and visit all leave nodes that intersect with the ray. All the access to other nodes are considered as overhead. The rope structure reduces this overhead by providing links to neighboring nodes. During traversal, we first identify the entry point and the exit point of the ray on the root node's bounding box that bounds the entire object. Call this exit point the root exit point. Then we travese to and access the leaf node that intersects the entry point. We update the entry point to be the exit point of the ray intersecting with the current leave node. The above process is repeated until the entry point is updated to be the same with the root exit point. A more detailed explanation can be found in the paper.
## Performance
Time of Kd-tree construct for different objects and different bounds

Naive traversal vs KD-tree traversal

CPU KD-tree traversal vs GPU KD-tree traversal

Stream Compaction vs no Stream Compaction
## Result Demo
Pictures
