#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <device_launch_parameters.h>

#include "Vector.h"

class Transform
{
public:
	// Constructors
	Transform()
	{
		this->_position = Vector3();
		this->_rotation = Vector3();
		this->_scale = Vector3(1.f, 1.f, 1.f);
	}

	// Getters and setters
	__host__ __device__ Vector3 GetPosition() const { return this->_position; }
	__host__ __device__ Vector3 GetRotation() const { return this->_rotation; }
	__host__ __device__ Vector3 GetScale() const { return this->_scale; }
	void SetPosition(const Vector3 &v3) { this->_position = v3; }
	void SetRotation(const Vector3 &v3) { this->_rotation = v3; }
	void SetScale(const Vector3 &v3) { this->_scale = v3; }

	// Methods
	void Translate(const Vector3 &v3) { this->_position += v3; }
	__host__ __device__ Vector3 TransformPoint(const Vector3 &v3);
	__host__ __device__ Vector3 TransformDirection(const Vector3 &v3);

protected:
	Vector3 _position, _rotation, _scale;
};

#endif