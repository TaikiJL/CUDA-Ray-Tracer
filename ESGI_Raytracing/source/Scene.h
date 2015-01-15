#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include "Geometry.h"
#include "Light.h"
#include "Color.h"
#include "Camera.h"
#include "Bitmap.h"

class Scene
{
public:
	inline static int GetRenderWidth() { return _width; }
	inline static int GetRenderHeight() { return _height; }
	inline static void SetRenderResolution(const int width, const int height) { _width = width; _height = height; }
	inline static unsigned short int GetMaxBounce() { return _maxBounce; }
	inline static void SetMaxBounce(unsigned short int bounce) { _maxBounce = bounce; }
	inline static std::vector<Geometry>& GetGeometries() { return _geometries; }
	inline static std::vector<Light>& GetLights() { return _lights; }
	inline static std::vector<Camera>& GetCameras() { return _cameras; }
	inline static Color GetAmbientColor() { return _ambientColor; }
	inline static void SetAmbientColor(const Color &color) { _ambientColor = color; }
	inline static Camera& GetMainCamera() { return _mainCamera; }
	inline static void SetMainCamera(const Camera &camera) { _mainCamera = camera; }
	inline static std::vector<Bitmap>& GetTextures() { return _textures; }

private:
	static int _width;
	static int _height;
	static unsigned short int _maxBounce;
	static std::vector<Geometry> _geometries;
	static std::vector<Light> _lights;
	static std::vector<Camera> Scene::_cameras;
	static Color _ambientColor;
	static Camera _mainCamera;
	static std::vector<Bitmap> _textures;
};

#endif