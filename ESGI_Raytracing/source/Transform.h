#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <device_launch_parameters.h>

#include "Vector.h"
#include "Matrix.h"
#include "MyMath.h"

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
	__host__ __device__ Vector3 TransformPoint(const Vector3 &v3)
	{
		return v3 + this->GetPosition();
	}
	__host__ __device__ Vector3 TransformDirection(const Vector3 &v3)
	{
		float xAngle = this->_rotation.x * Deg2Rad();
		float yAngle = this->_rotation.y * Deg2Rad();
		float zAngle = this->_rotation.z * Deg2Rad();
		Matrix4x3 rotationMatrix(cosf(zAngle) * cosf(yAngle), sinf(zAngle) * cosf(xAngle) + cosf(zAngle) * sinf(yAngle) * sinf(xAngle), sinf(zAngle) * sinf(xAngle) - cosf(zAngle) * sinf(yAngle) * cosf(xAngle), 0.f,
			-sinf(zAngle) * cosf(yAngle), cosf(zAngle) * cosf(xAngle) + sinf(zAngle) * sinf(yAngle) * sinf(xAngle), cosf(zAngle) * sinf(xAngle) + sinf(zAngle) * sinf(yAngle) * cosf(xAngle), 0.f,
			sinf(yAngle), -cosf(yAngle) * sinf(xAngle), cosf(yAngle) * cosf(xAngle), 0.f);

		return rotationMatrix * Vector4(v3, 0.f);
	}
	__host__ __device__ Vector3 InverseTransformPoint(const Vector3 &v3)
	{
		return v3 - this->GetPosition();
	}
	__host__ __device__ Vector3 InverseTransformDirection(const Vector3 &v3)
	{
		float xAngle = -1.f * this->_rotation.x * Deg2Rad();
		float yAngle = -1.f * this->_rotation.y * Deg2Rad();
		float zAngle = -1.f * this->_rotation.z * Deg2Rad();
		Matrix4x3 rotationMatrix(cosf(zAngle) * cosf(yAngle), sinf(zAngle) * cosf(xAngle) + cosf(zAngle) * sinf(yAngle) * sinf(xAngle), sinf(zAngle) * sinf(xAngle) - cosf(zAngle) * sinf(yAngle) * cosf(xAngle), 0.f,
			-sinf(zAngle) * cosf(yAngle), cosf(zAngle) * cosf(xAngle) + sinf(zAngle) * sinf(yAngle) * sinf(xAngle), cosf(zAngle) * sinf(xAngle) + sinf(zAngle) * sinf(yAngle) * cosf(xAngle), 0.f,
			sinf(yAngle), -cosf(yAngle) * sinf(xAngle), cosf(yAngle) * cosf(xAngle), 0.f);

		return rotationMatrix * Vector4(v3, 0.f);
	}

	// Static Functions
	__host__ __device__ static Vector4 RotateX(float angle, const Vector4 &v4)
	{
		float radAngle = angle * Deg2Rad();
		Matrix4x4 rotationMatrix = Matrix4x4(1.f, 0.f, 0.f, 0.f,
			0.f, cosf(radAngle), sinf(radAngle), 0.f,
			0.f, -sinf(radAngle), cosf(radAngle), 0.f,
			0.f, 0.f, 0.f, 1.f);

		return rotationMatrix * v4;
	}
	__host__ __device__ static Vector4 RotateY(float angle, const Vector4 &v4)
	{
		float radAngle = angle * Deg2Rad();
		Matrix4x4 rotationMatrix = Matrix4x4(cosf(radAngle), 0.f, -sinf(radAngle), 0.f,
			0.f, 1.f, 0.f, 0.f,
			sinf(radAngle), 0.f, cosf(radAngle), 0.f,
			0.f, 0.f, 0.f, 1.f);

		return rotationMatrix * v4;
	}
	__host__ __device__ static Vector4 RotateZ(float angle, const Vector4 &v4)
	{
		float radAngle = angle * Deg2Rad();
		Matrix4x4 rotationMatrix = Matrix4x4(cosf(radAngle), sinf(radAngle), 0.f, 0.f,
			-sinf(radAngle), cosf(radAngle), 0.f, 0.f,
			0.f, 0.f, 1.f, 0.f,
			0.f, 0.f, 0.f, 1.f);

		return rotationMatrix * v4;
	}

protected:
	Vector3 _position, _rotation, _scale;
};

#endif