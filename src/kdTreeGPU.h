#pragma once

#include "kdTree.h"
#include "interactions.h"
#include "intersections.h"
__host__ __device__ int idxMax(int x, int y) {
	if (x > y)
		return x;
	else
		return y;
}


__host__ __device__ float floatMax(float x, float y) {
	if (x > y)
		return x;
	else
		return y;
}

__host__ __device__ float floatMin(float x, float y) {
	if (x < y)
		return x;
	else
		return y;
}
__host__ __device__ bool intersectAABB(Ray r, BoundingBox & bbox, float& t_entry, float& t_exit) {
	auto ray_o = r.origin;
	auto ray_dir = r.direction;
	glm::vec3 dirfrac(1.0f / ray_dir.x, 1.0f / ray_dir.y, 1.0f / ray_dir.z);

	float t1 = (bbox.min.x - ray_o.x) * dirfrac.x;
	float t2 = (bbox.max.x - ray_o.x) * dirfrac.x;
	float t3 = (bbox.min.y - ray_o.y) * dirfrac.y;
	float t4 = (bbox.max.y - ray_o.y) * dirfrac.y;
	float t5 = (bbox.min.z - ray_o.z) * dirfrac.z;
	float t6 = (bbox.max.z - ray_o.z) * dirfrac.z;

	t_entry = floatMax(floatMax(floatMin(t1, t2), floatMin(t3, t4)), floatMin(t5, t6));
	t_exit = floatMin(floatMin(floatMax(t1, t2), floatMax(t3, t4)), floatMax(t5, t6));

	// If tmax < 0, ray intersects AABB, but entire AABB is behind ray, so reject.
	if (t_exit < 0.0f) {
		return false;
	}

	// If tmin > tmax, ray does not intersect AABB.
	if (t_entry > t_exit) {
		return false;
	}

	return true;
}

__host__ __device__ bool onLeft(glm::vec3 p, KdTreeNode& node) {
	if (node.split_plane_axis == X_AXIS) {
		return (p.x < node.split_plane_value);
	}
	else if (node.split_plane_axis == Y_AXIS) {
		return (p.y < node.split_plane_value);
	}
	else if (node.split_plane_axis == Z_AXIS) {
		return (p.z < node.split_plane_value);
	}
	// Something went wrong because split_plane_axis is not set to one of the three allowed values.
	else {
		return false;
	}
}

__host__ __device__ int getNeighborIdx(glm::vec3 p, KdTreeNode& node) {
	const float GPU_KD_TREE_EPSILON = 0.00001f;

	float faceDis[6];
	faceDis[LEFT] = fabsf(p.x - node.bb.min.x);
	faceDis[TOP] = fabsf(p.z - node.bb.max.z);
	faceDis[RIGHT] = fabsf(p.x - node.bb.max.x);
	faceDis[BOTTOM] = fabsf(p.z - node.bb.min.z);
	faceDis[FRONT] = fabsf(p.y - node.bb.max.y);
	faceDis[BACK] = fabsf(p.y - node.bb.min.y);

	for (int i = 0; i < 6; i++) {
		bool flag = true;
		for (int j = 0; j < 6; j++) {
			if (faceDis[i] > faceDis[j]) {
				flag = false;
				break;
			}
		}
		if (flag)
			return node.neighbor_node_indices[i];
	}

	return -1;
	printf("ERROR\n");
	/*
	// Check left face.
	if (fabsf(p.x - node.bb.min.x) < GPU_KD_TREE_EPSILON) {
		return node.neighbor_node_indices[LEFT];
	}
	// Check front face.
	else if (fabsf(p.z - node.bb.max.z) < GPU_KD_TREE_EPSILON) {
		return node.neighbor_node_indices[FRONT];
	}
	// Check right face.
	else if (fabsf(p.x - node.bb.max.x) < GPU_KD_TREE_EPSILON) {
		return node.neighbor_node_indices[RIGHT];
	}
	// Check back face.
	else if (fabsf(p.z - node.bb.min.z) < GPU_KD_TREE_EPSILON) {
		return node.neighbor_node_indices[BACK];
	}
	// Check top face.
	else if (fabsf(p.y - node.bb.max.y) < GPU_KD_TREE_EPSILON) {
		return node.neighbor_node_indices[TOP];
	}
	// Check bottom face.
	else if (fabsf(p.y - node.bb.min.y) < GPU_KD_TREE_EPSILON) {
		return node.neighbor_node_indices[BOTTOM];
	}
	// p should be a point on one of the faces of this node's bounding box, but in this case, it isn't.
	else {
		return -1;
	}*/
}