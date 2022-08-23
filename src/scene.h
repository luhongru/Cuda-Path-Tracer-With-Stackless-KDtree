#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "kdTree.h"

using namespace std;
class Scene {
private:
    std::ifstream fp_in;
    int loadMaterial(std::string materialid);
    int loadGeom(std::string objectid);
    int loadCamera();
public:
    Scene(std::string filename);
    ~Scene() {}

    int loadObj(std::string filename);
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    KdTree * kdTree;
};
