# Cuda Path Tracer With Stackless KDtree
This repository implements a path tracer with tracer and a CPU contructed, GPU traversed KD-tree acceleration structure. The path tracer uses cuda kernels to shade the images and openGL to render.

## Stackless KD-Tree
KD-Tree is an efficient acceleration structure for static scenes in ray tracing, and GPUs provide great performance in large throughput parallel computation. However, an intuitive contruction of KD-Tree requires a recursive traversal (or stack-based) which is poorly supported on GPUs. In order to exploit the computation power of GPUs, multiple designs of stackless KD-tree are proposed. This project implements a stackless Kd-tree presented by Stefan et al. which augment the tree nodes with "ropes". A tree node has one rope for each of face of its bounding box. And the rope connects to a neighboring node on that face and the corresponding face of the neighboring node must be larger. A figure from the paper by Stefan provides a good illustration here.   
![alt text](./img/ropes.PNG?raw=true)   
Looking at the right face of node 2 (with blue rope), it is adjacent to both node 4 and node 6. Therefore, this blue rope must point to a common ancestor node of node 4 and node 6, which is the blue node in the figure.

The goal of traversing a KD-tree is to find and visit all leave nodes that intersect with the ray. All the access to other nodes are considered as overhead. The rope structure reduces this overhead by providing links to neighboring nodes. During traversal, we first identify the entry point and the exit point of the ray on the root node's bounding box that bounds the entire object. Call this exit point the root exit point. Then we travese to and access the leaf node that intersects the entry point. We update the entry point to be the exit point of the ray intersecting with the current leave node. The above process is repeated until the entry point is updated to be the same with the root exit point. A more detailed explanation can be found in the paper.
## Performance
The performance boost of using KD-tree is significant. The following figure shows the runtime of the intersection kerenl using KD-tree vs naively traversing all triangles. We can observe around 8x faster performance.   
![alt text](./profile/naive_kdtree.png?raw=true)   
An important parameter of a KD-tree is the upper bound of the size of a leaf node, e.g. the largest number of triangles a leaf node can contain. This bound affects the construction time of the KD-tree, which is a one-time overhead, and the run time of the intersection kernel, which is computed repeatedly. The following figure shows the trade off. We can see that when the bound is smaller, it takes longer time to construct the tree and short time to traverse the tree. Given that the contruction only happens at initialization, we should pick the smallest bound with acceptable construction time.     
![alt text](./profile/bound.png?raw=true)     
The next figure shows the construction time of the KD-tree with different bounds and number of triangles.    
![alt text](./profile/cons_tris.png?raw=true)    
This implementation also utilize stream compaction to boost performance. Each ray has an iteration depth of 10. Stream compaction excludes rays that has already completed from computation. The following figure shows the runtime of one computation iteration with and without stream compaction (the overhead of stream compation itself is included).  
![alt text](./profile/par_no.png?raw=true)  
## Result Demo
A example run of a cessna toy plane with 7446 triangles.  

51 iterations:  
![alt text](./img/51.png?raw=true)  
1487 iterations:  
![alt text](./img/1487.png?raw=true)  
## Build & Run
The implementation is built and tested with VS2019 and CUDA 11.7
* Clone the repository
* Configure the project using CMake. Select VS2019 as generator and x64 as platform. Generate the project
* Open the project with VS2019
* Build in Release mode
* Go to directory ./profile and run default_run.py inside this directory
The script should give a sample run of a toy cessna plane. You can modify the parameters used in the script
