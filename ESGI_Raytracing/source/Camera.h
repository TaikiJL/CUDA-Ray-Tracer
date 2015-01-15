#ifndef CAMERA_H
#define CAMERA_H

#include "Transform.h"
#include "MyMath.h"
#include "Color.h"

enum FOVType
{
	HORIZONTAL_FOV,
	VERTICAL_FOV,
	FOCAL_LENGTH
};

class Camera : public Transform
{
public:
	// Constructors
	Camera();
	
	// Getters & Setters
	float GetFocalLength() const { return this->_focalLength; }
	float GetHorizontalFOV() const { return this->_horizontalFOV; }
	float GetVerticalFOV() const { return this->_verticalFOV; }
	float GetHorizontalHalfFOV() const { return this->_horizontalFOV * 0.5f * Deg2Rad(); }
	float GetVerticalHalfFOV() const { return this->_verticalFOV * 0.5f * Deg2Rad(); }
	__host__ __device__ Color GetBackgroundColor() const { return this->_backgroundColor; }
	void SetBackgroundColor(const Color color) { this->_backgroundColor = color; }
	void SetFocalLength(const float length);
	void SetHorizontalFOV(const float fov);
	void SetVerticalFOV(const float fov);

private:
	float _focalLength, _horizontalFOV, _verticalFOV;
	FOVType _lastSettedFOV;
	Color _backgroundColor;
};

#endif