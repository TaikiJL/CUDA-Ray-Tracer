#ifndef LIGHT_H
#define LIGHT_H

#include "Transform.h"
#include "Color.h"

enum LightType
{
	OMNI,
	SPOT,
	DIRECTIONAL
};

class Light : public Transform
{
public:
	// Constructors
	Light();

	// Getters & Setters
	__host__ __device__ float GetIntensity() const { return this->_intensity; }
	__host__ __device__ LightType GetLightType() const { return this->_lightType; }
	__host__ __device__ Color GetLightColor() const { return this->_color; }
	void SetIntensity(const float intensity) { this->_intensity = intensity; }
	void SetLightType(LightType lightType) { this->_lightType = lightType; }
	void SetLightColor(const Color &color) { this->_color = color; }

private:
	float _intensity;
	LightType _lightType;
	Color _color;
};

#endif