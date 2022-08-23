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
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "kdTreeGPU.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene* hst_scene = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Triangle * dev_tris = NULL;
static KdTreeNode* dev_treeNodes = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

static PathSegment* hst_paths = NULL;
static ShadeableIntersection* hst_intersections = NULL;
static int pixelcount = 0;
// TODO: static variables for device memory, any extra info you need, etc
// ...


void kdTreeIntersectionSingleRayCPU(KdTreeNode* nodes, int num_path, Geom* geoms, int geomIdx, PathSegment* pathSegments, Triangle* tris, ShadeableIntersection* intersections, int nillIdx, Scene* hst_scene, int path_index);

void debug(Scene* scene) {
	PathSegment ps;
	ps.ray.direction.x = -0.850967;
	ps.ray.direction.y = -0.485993;
	ps.ray.direction.z = -0.199163;
	ps.ray.origin.x = 0.540807;
	ps.ray.origin.y = 4.879376;
	ps.ray.origin.z = 1.215921;
	//kdTreeIntersectionSingleRayCPU(scene->kdTree->treeNodes.data(), &scene->kdTree->geom, 0, ps, scene->kdTree->tris.data(), -1);

}
void pathtraceInit(Scene* scene) {
	hst_scene = scene;
	//debug(scene);
	const Camera& cam = hst_scene->state.camera;
	pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_tris, scene->kdTree->tris.size() * sizeof(Triangle));
	cudaMemcpy(dev_tris, scene->kdTree->tris.data(), scene->kdTree->tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_treeNodes, scene->kdTree->treeNodes.size() * sizeof(KdTreeNode));
	cudaMemcpy(dev_treeNodes, scene->kdTree->treeNodes.data(), scene->kdTree->treeNodes.size() * sizeof(KdTreeNode), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need
	hst_paths = new PathSegment[pixelcount * sizeof(PathSegment)];
	hst_intersections = new ShadeableIntersection[pixelcount * sizeof(ShadeableIntersection)];
	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_tris);
	cudaFree(dev_treeNodes);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	free(hst_paths);
	free(hst_intersections);
	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__global__ void kdTreeIntersection(KdTreeNode* nodes, int num_path, Geom* geoms, int geomIdx, PathSegment* pathSegments, Triangle* tris, ShadeableIntersection* intersections, int nillIdx) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int currNodes[BLOCK_SIZE_1D];
	__shared__ int array[BLOCK_SIZE_1D / 2];
	int& currentN = currNodes[threadIdx.x];
	currentN = 0;
	int& traversedN = array[0];
	traversedN = 0;
	if (path_index >= num_path)
		currentN = nillIdx;

	PathSegment& pathSegment = pathSegments[path_index];
	float t_entry, t_exit;

	glm::vec3 ro = multiplyMV(geoms[geomIdx].inverseTransform, glm::vec4(pathSegment.ray.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(geoms[geomIdx].inverseTransform, glm::vec4(pathSegment.ray.direction, 0.0f)));

	Ray rt;
	rt.origin = ro;
	rt.direction = rd;

	intersectAABB(rt, nodes[0].bb, t_entry, t_exit);

	glm::vec3 p_entry, p_exit;
	p_entry = getPointOnRay(rt, t_entry);
	p_exit = getPointOnRay(rt, t_exit);

	float t = intersections[path_index].t;;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	float t_min = t;
	int hit_geom_index = -1;
	bool outside = true;

	
	while (true) {
		//check currNode is done
		if (t_entry >= t_exit)
			currentN = nillIdx;

		//calculate traversedN
		int stride = BLOCK_SIZE_1D / 2;
		__syncthreads();
		if (threadIdx.x < stride)
			array[threadIdx.x] = idxMax(currNodes[threadIdx.x], currNodes[threadIdx.x + stride]);
		__syncthreads();
		for (stride = BLOCK_SIZE_1D / 4; stride >= 1; stride /= 2) {
			if (threadIdx.x < stride)
				array[threadIdx.x] = idxMax(array[threadIdx.x], array[threadIdx.x + stride]);
			__syncthreads();
		}
		
		//end condition
		if (traversedN == nillIdx)
			break;

		//down traversal loop
		while (!nodes[traversedN].isLeaf) {
			//update currentN
			int flag = 0;
			if (currentN == traversedN) {
				if (onLeft(p_entry, nodes[currentN])) {
					currentN = nodes[currentN].leftNodeIdx;
					flag = 1;
				}
				else {
					currentN = nodes[currentN].rightNodeIdx;
					flag = -1;
				}
					
			}

			//update traversedN
			__syncthreads();
			if (flag != 0) {
				traversedN = currentN;
			}
			__syncthreads();
				
			
		}
		//intersect
		if (currentN == traversedN) {
			glm::vec3 tmp_intersect;
			glm::vec3 tmp_normal;
			bool outside = true;
			
			for (int i = 0; i < nodes[currentN].triCount; i++) {
				int triIdx = i + nodes[currentN].triStartIdx;
				auto tri = tris[triIdx];
				t = triangleIntersectionTest(geoms[geomIdx], tri, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				
				if ((t > 0.0f && t_min > t) || t_min == -1)
				{
					t_min = t;
					hit_geom_index = geomIdx;
					intersect_point = tmp_intersect;
					normal = tmp_normal;
				}

			}

			intersectAABB(rt, nodes[currentN].bb, t_entry, t_exit);
			p_entry = getPointOnRay(rt, t_entry);
			p_exit = getPointOnRay(rt, t_exit);
			currentN = getNeighborIdx(p_exit, nodes[currentN]);
		}
	}


	if (hit_geom_index != -1) {
		intersections[path_index].t = t_min;
		intersections[path_index].materialId = geoms[hit_geom_index].materialid;
		intersections[path_index].surfaceNormal = normal;
		intersections[path_index].intersectPoint = intersect_point;
	}
}

__global__ void kdTreeIntersectionSingleRay(KdTreeNode* nodes, int num_path, Geom* geoms, int geomIdx, PathSegment* pathSegments, Triangle* tris, ShadeableIntersection* intersections, int nillIdx) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index > num_path)
		return;

	PathSegment& pathSegment = pathSegments[path_index];
	int currNode = 0;
	float t_entry, t_exit;

	glm::vec3 ro = multiplyMV(geoms[geomIdx].inverseTransform, glm::vec4(pathSegment.ray.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(geoms[geomIdx].inverseTransform, glm::vec4(pathSegment.ray.direction, 0.0f)));

	Ray rt;
	rt.origin = ro;
	rt.direction = rd;

	if (!intersectAABB(rt, nodes[0].bb, t_entry, t_exit))
		return;

	glm::vec3 p_entry, p_exit;

	float t = intersections[path_index].t;;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	float t_min = t;
	int hit_geom_index = -1;
	bool outside = true;
	int count = 0;
	while (t_entry < t_exit && currNode != -1 && count < 20) {
		count++;
		p_entry = getPointOnRay(rt, t_entry);
		while (!nodes[currNode].isLeaf) {
			if (onLeft(p_entry, nodes[currNode]))
				currNode = nodes[currNode].leftNodeIdx;
			else
				currNode = nodes[currNode].rightNodeIdx;
		}


		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		bool outside = true;

		for (int i = 0; i < nodes[currNode].triCount; i++) {
			int triIdx = i + nodes[currNode].triStartIdx;
			auto tri = tris[triIdx];
			t = triangleIntersectionTest(geoms[geomIdx], tri, pathSegment.ray, tmp_intersect, tmp_normal, outside);

			if ((t > 0.0f && t_min > t) || t_min == -1)
			{
				t_min = t;
				hit_geom_index = geomIdx;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}

		}

		float temp_t = 0;
		intersectAABB(rt, nodes[currNode].bb, temp_t, t_entry);
		
		p_exit = getPointOnRay(rt, t_entry);
		currNode = getNeighborIdx(p_exit, nodes[currNode]);
	}

	if (hit_geom_index != -1) {
		intersections[path_index].t = t_min;
		intersections[path_index].materialId = geoms[hit_geom_index].materialid;
		intersections[path_index].surfaceNormal = normal;
		intersections[path_index].intersectPoint = intersect_point;
	}
}

void kdTreeIntersectionSingleRayCPU(KdTreeNode* nodes, int num_path, Geom* geoms, int geomIdx, PathSegment * pathSegments, Triangle* tris, ShadeableIntersection* intersections, int nillIdx, Scene * hst_scene, int path_index) {
	if (path_index > num_path)
		return;

	PathSegment& pathSegment = pathSegments[path_index];
	int currNode = 0;
	float t_entry, t_exit;

	glm::vec3 ro = multiplyMV(geoms[geomIdx].inverseTransform, glm::vec4(pathSegment.ray.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(geoms[geomIdx].inverseTransform, glm::vec4(pathSegment.ray.direction, 0.0f)));

	Ray rt;
	rt.origin = ro;
	rt.direction = rd;

	if (!intersectAABB(rt, nodes[0].bb, t_entry, t_exit))
		return;

	glm::vec3 p_entry, p_exit;

	float t = intersections[path_index].t;;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	float t_min = t;
	int hit_geom_index = -1;
	bool outside = true;

	int count = 0;

	while (t_entry < t_exit && currNode != -1) {
		count++;
		printf("%d\n", count);
		p_entry = getPointOnRay(rt, t_entry);
		while (!nodes[currNode].isLeaf) {
			if (onLeft(p_entry, nodes[currNode]))
				currNode = nodes[currNode].leftNodeIdx;
			else
				currNode = nodes[currNode].rightNodeIdx;
		}


		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		bool outside = true;

		for (int i = 0; i < nodes[currNode].triCount; i++) {
			int triIdx = i + nodes[currNode].triStartIdx;
			auto tri = tris[triIdx];
			t = triangleIntersectionTest(geoms[geomIdx], tri, pathSegment.ray, tmp_intersect, tmp_normal, outside);

			if ((t > 0.0f && t_min > t) || t_min == -1)
			{
				t_min = t;
				hit_geom_index = geomIdx;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}

		}

		float temp_t = 0;
		intersectAABB(rt, nodes[currNode].bb, temp_t, t_entry);

		p_exit = getPointOnRay(rt, t_entry);
		currNode = getNeighborIdx(p_exit, nodes[currNode]);
	}

	if (hit_geom_index != -1) {
		intersections[path_index].t = t_min;
		intersections[path_index].materialId = geoms[hit_geom_index].materialid;
		intersections[path_index].surfaceNormal = normal;
		intersections[path_index].intersectPoint = intersect_point;
	}
	else {
		for (int i = 0; i < hst_scene->kdTree->tris.size(); i++) {
			glm::vec3 tmp_intersect;
			glm::vec3 tmp_normal;
			bool outside = true;
			t = triangleIntersectionTest(geoms[geomIdx], tris[i], pathSegment.ray, tmp_intersect, tmp_normal, outside);

			if ((t > 0.0f && t_min > t) || t_min == -1)
			{
				t_min = t;
				hit_geom_index = geomIdx;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}
		
		if (hit_geom_index != -1) {
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].intersectPoint = intersect_point;
		}
	}
}

__global__ void kdTreeIntersectionNaive(KdTreeNode* nodes, int num_path, Geom* geoms, int geomIdx, PathSegment* pathSegments, Triangle* tris, int num_tri, ShadeableIntersection* intersections, int nillIdx) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index > num_path)
		return;

	PathSegment& pathSegment = pathSegments[path_index];
	int currNode = 0;
	float t_entry, t_exit;

	glm::vec3 ro = multiplyMV(geoms[geomIdx].inverseTransform, glm::vec4(pathSegment.ray.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(geoms[geomIdx].inverseTransform, glm::vec4(pathSegment.ray.direction, 0.0f)));

	Ray rt;
	rt.origin = ro;
	rt.direction = rd;

	if (!intersectAABB(rt, nodes[0].bb, t_entry, t_exit))
		return;

	glm::vec3 p_entry, p_exit;

	float t = intersections[path_index].t;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	float t_min = t;
	int hit_geom_index = -1;
	bool outside = true;

	for (int i = 0; i < num_tri; i++) {
		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		bool outside = true;
		t = triangleIntersectionTest(geoms[geomIdx], tris[i], pathSegment.ray, tmp_intersect, tmp_normal, outside);

		if ((t > 0.0f && t_min > t) || t_min == -1)
		{
			t_min = t;
			hit_geom_index = geomIdx;
			intersect_point = tmp_intersect;
			normal = tmp_normal;
		}
	}

	if (hit_geom_index != -1) {
		intersections[path_index].t = t_min;
		intersections[path_index].materialId = geoms[hit_geom_index].materialid;
		intersections[path_index].surfaceNormal = normal;
		intersections[path_index].intersectPoint = intersect_point;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				//t = -1;
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				//t = -1;
			}
			else if (geom.type == OBJECT) {
				glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(pathSegment.ray.origin, 1.0f));
				glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(pathSegment.ray.direction, 0.0f)));

				Ray rt;
				rt.origin = ro;
				rt.direction = rd;
				float t1, t2;
				bool aabb = intersectAABB(rt, geom.bb, t1, t2);
				pathSegments[path_index].intersectWithAABB = aabb;
				intersections[path_index].intersectWithAABB = aabb;
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].intersectPoint = intersect_point;
		}
	}
}



// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance * 0.5f);

			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				//pathSegments[idx].color *= materialColor * 0.8f;
				//pathSegments[idx].color *= u01(rng); // apply some noise because why not
				//pathSegments[idx].ray.origin = intersection.intersectPoint + 0.01f * intersection.surfaceNormal;
				//pathSegments[idx].ray.direction = 2.0f * intersection.surfaceNormal * glm::dot(-intersection.surfaceNormal, pathSegments[idx].ray.direction) + pathSegments[idx].ray.direction;

				//glm::vec3 rndPoint = randomUnitBall(iter, idx);
				//rndPoint += intersection.surfaceNormal;
				//pathSegments[idx].ray.direction = glm::normalize(rndPoint);
				scatterRay(pathSegments[idx], intersection.intersectPoint, intersection.surfaceNormal, material, rng);
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
		}
	}
}


// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) {
			//printf("intersect t %f\n", intersection.t);
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance * 0.5f);
				pathSegments[idx].remainingBounces--;
			}
			
			else {
				scatterRay(pathSegments[idx], intersection.intersectPoint, intersection.surfaceNormal, material, rng);
				pathSegments[idx].remainingBounces--;
			}
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

struct comparator {
	__host__ __device__ bool operator() (const ShadeableIntersection& lhs, const ShadeableIntersection& rhs) {
		return lhs.materialId < rhs.materialId;
	}
};

struct is_complete {
	__host__ __device__ bool operator() (const PathSegment& p) {
		return p.remainingBounces > 0;
	}
};

struct is_aabb_path {
	__host__ __device__ bool operator() (const PathSegment& p) {
		return p.intersectWithAABB;
	}
};

struct is_aabb_inter {
	__host__ __device__ bool operator() (const ShadeableIntersection& p) {
		return p.intersectWithAABB;
	}
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {
		
		PathSegment* new_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, is_complete());
		num_paths = new_end - dev_paths;
		if (num_paths < 1) break;
	

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
		checkCUDAError("trace one bounce");
		
		PathSegment* new_end_path = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, is_aabb_path());
		thrust::partition(thrust::device, dev_intersections, dev_intersections + num_paths, is_aabb_inter());
		int num_paths_aabb = new_end_path - dev_paths;
		dim3 numblocksAABB = (num_paths_aabb + blockSize1d - 1) / blockSize1d;
		/*
		cudaMemcpy(hst_paths, dev_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToHost);
		cudaMemcpy(hst_intersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToHost);

		for (int i = 0; i < pixelcount; i++) {
			kdTreeIntersectionSingleRayCPU(hst_scene->kdTree->treeNodes.data(), num_paths_aabb, &hst_scene->kdTree->geom, 0, hst_paths, hst_scene->kdTree->tris.data(), hst_intersections, -1, hst_scene, i);
		}
		//kdTreeIntersectionSingleRay << <numblocksAABB, blockSize1d >> > (dev_treeNodes, num_paths_aabb, dev_geoms, 6, dev_paths, dev_tris, dev_intersections, -1);

		cudaMemcpy(dev_paths, hst_paths, pixelcount * sizeof(PathSegment), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_intersections, hst_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyHostToDevice);*/
		
		kdTreeIntersectionSingleRay << <numblocksAABB, blockSize1d >> > (dev_treeNodes, num_paths_aabb, dev_geoms, 6, dev_paths, dev_tris, dev_intersections, -1);
		//kdTreeIntersection << <numblocksAABB, blockSize1d >> > (dev_treeNodes, num_paths_aabb, dev_geoms, 6, dev_paths, dev_tris, dev_intersections, -1);
		checkCUDAError("kd kernel");
		cudaDeviceSynchronize();
		depth++;
		printf("compute intersection done\n");

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.
		thrust::device_ptr<ShadeableIntersection> intersection_for_mID(dev_intersections);
		thrust::device_ptr<PathSegment> thrust_paths(dev_paths);

		thrust::sort_by_key(intersection_for_mID, intersection_for_mID + num_paths, thrust_paths, comparator());


		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);
		iterationComplete = depth == 5; // TODO: should be based off stream compaction results.
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
