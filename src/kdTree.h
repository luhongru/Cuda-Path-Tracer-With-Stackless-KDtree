#pragma once

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"


constexpr int BLOCK_SIZE_1D = 128;


enum Axis {
	X_AXIS,
	Y_AXIS,
	Z_AXIS,
};

enum AABBFace {
	LEFT,
	FRONT,
	RIGHT,
	BACK,
	TOP,
	BOTTOM,
};



struct KdTreeNode {
	//tree structure
	bool isLeaf;
	int leftNodeIdx;
	int rightNodeIdx;

	//triangle array (only in leaves)
	int triStartIdx;
	int triCount;

	//bounding box
	BoundingBox bb;
	Axis split_plane_axis;
	float split_plane_value;

	//ropes
	int neighbor_node_indices[6];
};

class KdTree {
	
	void readTriData(std::string filename, std::vector<Triangle>& tri_data);
	void setTreeGeom();
	void constructTree(std::vector<Triangle>& triDataArray);
	void connectRopes();
	void connectRopeRec(int nodeIdx, int neighbor_node_indices[6]);
public:
	std::vector<Triangle> tris;
	std::vector<KdTreeNode> treeNodes;
	Geom geom;

	KdTree(std::string filename);
	~KdTree();
};