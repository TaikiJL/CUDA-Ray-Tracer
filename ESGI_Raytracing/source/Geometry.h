#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "Vector.h"
#include "Transform.h"
#include "Material.h"

class Geometry : public Transform
{
public:
	Geometry() {}

	// Getters & Setters
	__host__ __device__ Material GetMaterial() { return this->_material; }
	void SetMaterialBaseColor(const Color &color) { this->_material.baseColor = color; }
	void SetMaterialShininess(const float shininess) { this->_material.shininess = shininess; }
	void SetMaterialSpecularColor(const Color &color) { this->_material.specularColor = color; }

	// Methods
	//__host__ __device__ virtual bool CheckRayIntersection(const Vector3 rayOrigin, const Vector3 rayDirection, Vector3 &intersectionPoint, Vector3 &intersectionNormal) = 0;

private:
	Material _material;
};

#endif