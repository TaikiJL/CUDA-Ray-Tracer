#ifndef SPHERE_H
#define SPHERE_H

#include "Geometry.h"

class Sphere : public Geometry
{
public:
	// Constructors
	Sphere() { this->_radius = 1.f; }
	Sphere(float radius) : _radius(radius) {}

	// Getters and setters
	float GetRadius() const { return this->_radius; }
	void SetRadius(const float r) { this->_radius = r; }

	// Methods
	__host__ __device__ bool CheckRayIntersection(const Vector3 rayOrigin, const Vector3 rayDirection, Vector3 &intersectionPoint, Vector3 &intersectionNormal)
	{
		Vector3 nRayDirection = Vector3::Normalized(rayDirection);
		float a = Vector3::Dot(this->_position - rayOrigin, nRayDirection);

		if (a <= 0.f)
			return false;

		float e = Vector3::Distance(this->_position, rayOrigin);

		float squareF = _radius * _radius - e * e + a * a;

		if (squareF < 0.f)
			return false;

		float t = a - sqrt(squareF);
		intersectionPoint = rayOrigin + nRayDirection * t;
		intersectionNormal = Vector3::Normalized(intersectionPoint - this->_position);

		return true;
	}

private:
	float _radius;
};

#endif