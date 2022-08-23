#include "kdTree.h"
#include <assert.h>
#include "tiny_obj_loader.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

constexpr int TRI_COUNT_UPPER_BOUND = 20;

KdTree::KdTree(std::string filename) {
	std::vector<Triangle> tri_data;
	readTriData(filename, tri_data);
	constructTree(tri_data);
	connectRopes();
	setTreeGeom();
}

KdTree::~KdTree() {}

void KdTree::readTriData(std::string filename, std::vector<Triangle>& tri_data) {
	tinyobj::ObjReaderConfig reader_config;
	reader_config.mtl_search_path = "./"; // Path to material files

	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(filename, reader_config)) {
		if (!reader.Error().empty()) {
			std::cerr << "TinyObjReader: " << reader.Error();
		}
		exit(1);
	}

	if (!reader.Warning().empty()) {
		std::cout << "TinyObjReader: " << reader.Warning();
	}

	auto& attrib = reader.GetAttrib();
	auto& shapes = reader.GetShapes();
	auto& materials = reader.GetMaterials();

	for (size_t s = 0; s < shapes.size(); s++) {
		// Loop over faces(polygon)
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

			Triangle tri;

			// Loop over vertices in the face.
			for (size_t v = 0; v < fv; v++) {
				// access to vertex
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

				glm::vec3& vertex = tri.v[v];
				vertex.x = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
				vertex.y = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
				vertex.z = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
			}

			glm::vec3 edge1 = tri.v[1] - tri.v[0];
			glm::vec3 edge2 = tri.v[2] - tri.v[0];
			tri.normal = glm::normalize(glm::cross(edge1, edge2));
			tri_data.push_back(tri);
			index_offset += fv;
		}
	}
}

void KdTree::setTreeGeom() {
	//hard code geom info
	geom.type = OBJECT;
	geom.materialid = 5;
	geom.translation.x = 0.0f;
	geom.translation.y = 4.5f;
	geom.translation.z = 4.0f;
	geom.rotation.x = 0.0f;
	geom.rotation.y = 40.0f;
	geom.rotation.z = 70.0f;
	geom.scale.x = 1.6f;
	geom.scale.y = 1.6f;
	geom.scale.z = 1.6f;
	geom.transform = utilityCore::buildTransformationMatrix(
		geom.translation, geom.rotation, geom.scale);
	geom.inverseTransform = glm::inverse(geom.transform);
	geom.invTranspose = glm::inverseTranspose(geom.transform);
	geom.bb = treeNodes[0].bb;
}

void setSplitPlane(KdTreeNode& node) {
	float xLength = node.bb.max.x - node.bb.min.x;
	float yLength = node.bb.max.y - node.bb.min.y;
	float zLength = node.bb.max.z - node.bb.min.z;

	if (xLength >= yLength && xLength >= zLength) {
		node.split_plane_axis = X_AXIS;
		node.split_plane_value = (node.bb.max.x + node.bb.min.x) / 2;
		return;
	}

	if (yLength >= xLength && yLength >= zLength) {
		node.split_plane_axis = Y_AXIS;
		node.split_plane_value = (node.bb.max.y + node.bb.min.y) / 2;
		return;
	}

	if (zLength >= yLength && zLength >= xLength) {
		node.split_plane_axis = Z_AXIS;
		node.split_plane_value = (node.bb.max.z + node.bb.min.z) / 2;
		return;
	}
	assert(node.split_plane_axis == X_AXIS || node.split_plane_axis == Y_AXIS || node.split_plane_axis == Z_AXIS);
}

BoundingBox getChildBoundingBox(KdTreeNode& parentNode, bool left) {
	BoundingBox bb = parentNode.bb;
	if (parentNode.split_plane_axis == X_AXIS) {
		if (left)
			bb.max.x = parentNode.split_plane_value;
		else
			bb.min.x = parentNode.split_plane_value;

		return bb;
	}
	if (parentNode.split_plane_axis == Y_AXIS) {
		if (left)
			bb.max.y = parentNode.split_plane_value;
		else
			bb.min.y = parentNode.split_plane_value;

		return bb;
	}
	if (parentNode.split_plane_axis == Z_AXIS) {
		if (left)
			bb.max.z = parentNode.split_plane_value;
		else
			bb.min.z = parentNode.split_plane_value;

		return bb;
	}
}

bool leftOfSplitPlane(KdTreeNode& node, Triangle& tri) {
	if (node.split_plane_axis == X_AXIS) {
		for (int i = 0; i < 3; i++) {
			if (tri.v[i].x < node.split_plane_value)
				return true;
		}
	}
	if (node.split_plane_axis == Y_AXIS) {
		for (int i = 0; i < 3; i++) {
			if (tri.v[i].y < node.split_plane_value)
				return true;
		}
	}
	if (node.split_plane_axis == Z_AXIS) {
		for (int i = 0; i < 3; i++) {
			if (tri.v[i].z < node.split_plane_value)
				return true;
		}
	}
	return false;
}

bool rightOfSplitPlane(KdTreeNode& node, Triangle& tri) {
	if (node.split_plane_axis == X_AXIS) {
		for (int i = 0; i < 3; i++) {
			if (tri.v[i].x > node.split_plane_value)
				return true;
		}
	}
	if (node.split_plane_axis == Y_AXIS) {
		for (int i = 0; i < 3; i++) {
			if (tri.v[i].y > node.split_plane_value)
				return true;
		}
	}
	if (node.split_plane_axis == Z_AXIS) {
		for (int i = 0; i < 3; i++) {
			if (tri.v[i].z > node.split_plane_value)
				return true;
		}
	}
	return false;
}

void KdTree::constructTree(std::vector<Triangle> &tri_data_array) {
	std::vector<std::vector<int>> tri_idx_array_of_nodes(1);
	for (int i = 0; i < tri_data_array.size(); i++) {
		tri_idx_array_of_nodes[0].push_back(i);
	}

	{
		KdTreeNode node;
		node.isLeaf = true;
		node.bb.min = tri_data_array[0].v[0];
		node.bb.max = tri_data_array[0].v[0];
		for (auto idx : tri_idx_array_of_nodes[0]) {
			for (int i = 0; i < 3; i++) {
				glm::vec3 v = tri_data_array[idx].v[i];
				node.bb.max.x = std::max(node.bb.max.x, v.x);
				node.bb.max.y = std::max(node.bb.max.y, v.y);
				node.bb.max.z = std::max(node.bb.max.z, v.z);

				node.bb.min.x = std::min(node.bb.min.x, v.x);
				node.bb.min.y = std::min(node.bb.min.y, v.y);
				node.bb.min.z = std::min(node.bb.min.z, v.z);
			}
		}
		treeNodes.push_back(node);
	}
	
	//begin contructing
	for (int i = 0; i < treeNodes.size(); i++) {
		auto& node = treeNodes[i];
		auto& tri_idx_array = tri_idx_array_of_nodes[i];
		if (!node.isLeaf || tri_idx_array.size() <= TRI_COUNT_UPPER_BOUND)
			continue;

		//tree structure
		node.isLeaf = false;
		node.leftNodeIdx = treeNodes.size();
		node.rightNodeIdx = treeNodes.size() + 1;

		//triangle array
		node.triStartIdx = -1;
		node.triCount = 0;

		//bounding box
		setSplitPlane(node);

		//ropes are left for connectRopes

		//Add new leaf nodes
		KdTreeNode leftNode;
		KdTreeNode rightNode;
		std::vector<int> leftNode_tri_idx_array;
		std::vector<int> rightNode_tri_idx_array;

		leftNode.isLeaf = true;
		rightNode.isLeaf = true;
		
		for (auto idx : tri_idx_array) {
			auto tri = tri_data_array[idx];
			if (leftOfSplitPlane(node, tri))
				leftNode_tri_idx_array.push_back(idx);
			if (rightOfSplitPlane(node, tri))
				rightNode_tri_idx_array.push_back(idx);
		}

		leftNode.bb = getChildBoundingBox(node, true);
		rightNode.bb = getChildBoundingBox(node, false);

		treeNodes.push_back(leftNode);
		treeNodes.push_back(rightNode);
		tri_idx_array_of_nodes.push_back(leftNode_tri_idx_array);
		tri_idx_array_of_nodes.push_back(rightNode_tri_idx_array);

		//tri_idx_array.clear();
	}

	assert(treeNodes.size() == tri_idx_array_of_nodes.size());
	//build tris vector and set triStartIdx, triCount
	for (int i = 0; i < treeNodes.size(); i++) {
		auto& node = treeNodes[i];
		auto& tri_idx_array = tri_idx_array_of_nodes[i];
		if (!node.isLeaf)
			continue;

		assert(tri_idx_array.size() <= TRI_COUNT_UPPER_BOUND);
		node.triStartIdx = tris.size();
		node.triCount = tri_idx_array.size();
		for (auto idx : tri_idx_array)
			tris.push_back(tri_data_array[idx]);
	}
}

void KdTree::connectRopes() {
	int nills[6] = { -1,-1,-1,-1,-1,-1 };
	connectRopeRec(0, nills);
}

void KdTree::connectRopeRec(int nodeIdx, int neighbor_node_indices[6]) {
	auto& node = treeNodes[nodeIdx];
	for (int i = 0; i < 6; i++)
		node.neighbor_node_indices[i] = neighbor_node_indices[i];

	if (node.isLeaf)
		return;

	AABBFace sl, sr;
	if (node.split_plane_axis == X_AXIS) {
		sl = LEFT;
		sr = RIGHT;
	}
	if (node.split_plane_axis == Y_AXIS) {
		sl = BACK;
		sr = FRONT;
	}
	if (node.split_plane_axis == Z_AXIS) {
		sl = BOTTOM;
		sr = TOP;
	}

	float v = node.split_plane_value;
	int rsLeft[6];
	for (int i = 0; i < 6; i++)
		rsLeft[i] = neighbor_node_indices[i];
	rsLeft[sr] = node.rightNodeIdx;
	connectRopeRec(node.leftNodeIdx, rsLeft);

	int rsRight[6];
	for (int i = 0; i < 6; i++)
		rsRight[i] = neighbor_node_indices[i];
	rsRight[sl] = node.leftNodeIdx;
	connectRopeRec(node.rightNodeIdx, rsRight);
}