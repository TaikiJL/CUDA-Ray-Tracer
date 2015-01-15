#include "Scene.h"

int Scene::_width = 1280;
int Scene::_height = 720;

unsigned short int Scene::_maxBounce = 0;

std::vector<Geometry> Scene::_geometries;
std::vector<Light> Scene::_lights;
std::vector<Camera> Scene::_cameras;
std::vector<Bitmap> Scene::_textures;

Color Scene::_ambientColor = Color(0.2f);
Camera Scene::_mainCamera;