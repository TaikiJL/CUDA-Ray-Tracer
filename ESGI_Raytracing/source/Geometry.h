#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "Vector.h"
#include "Transform.h"
#include "Mesh.h"
#include "Material.h"

class Geometry : public Transform
{
public:
	__host__ __device__ Geometry() {}
	
	// Getters & Setters
	__host__ __device__ Mesh& GetMesh() { return this->_mesh; }
	__host__ __device__ Material& GetMaterial() { return this->_material; }
	void SetMaterialBaseColor(const Color &color) { this->_material.baseColor = color; }
	void SetMaterialShininess(const float shininess) { this->_material.shininess = shininess; }
	void SetMaterialSpecularColor(const Color &color) { this->_material.specularColor = color; }

	// Methods
	__device__ bool CheckRayIntersection(Vector3 rayOrigin, Vector3 rayDirection, float &intersectionCoefficient, int &faceIndex)
	{
		rayOrigin = this->InverseTransformPoint(rayOrigin);
		rayDirection = this->InverseTransformDirection(rayDirection);

		return this->_mesh.CheckRayIntersection(rayOrigin, rayDirection, intersectionCoefficient, faceIndex);
	}
	void CreateSphereMesh(float radius, int segments);
	void CreateBoxMesh(float length, float width, float height);
	void CreatePlaneMesh(float length, float width);

private:
	Mesh _mesh;
	Material _material;
};

#endif