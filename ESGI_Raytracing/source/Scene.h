#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include "Geometry.h"
#include "Light.h"
#include "Color.h"
#include "Sphere.h"
#include "Camera.h"

struct Scene
{
public:
	inline static int GetRenderWidth() { return _width; }
	inline static int GetRenderHeight() { return _height; }
	inline static void SetRenderResolution(const int width, const int height) { _width = width; _height = height; }
	inline static void AddGeometry(Sphere geometryAdress) { _geometries.push_back(geometryAdress); }
	inline static std::vector<Sphere>& GetGeometries() { return _geometries; }
	inline static void AddLight(Light light) { _lights.push_back(light); }
	inline static std::vector<Light>& GetLights() { return _lights; }
	inline static Color GetAmbientColor() { return _ambientColor; }
	inline static void SetAmbientColor(const Color &color) { _ambientColor = color; }
	inline static Camera& GetMainCamera() { return _mainCamera; }
	inline static void SetMainCamera(const Camera &camera) { _mainCamera = camera; }

private:
	static int _width;
	static int _height;
	static std::vector<Sphere> _geometries;
	static std::vector<Light> _lights;
	static Color _ambientColor;
	static Camera _mainCamera;
};

#endif