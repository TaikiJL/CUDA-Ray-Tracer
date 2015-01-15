#include "Scene.h"

int Scene::_width = 1280;
int Scene::_height = 720;

std::vector<Sphere> Scene::_geometries;
std::vector<Light> Scene::_lights;

Color Scene::_ambientColor = Color(0.2f);
Camera Scene::_mainCamera;