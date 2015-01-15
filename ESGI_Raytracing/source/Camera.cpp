#include "Camera.h"

#include <cmath>

#include "Scene.h"
#include "MyMath.h"

Camera::Camera()
{
	this->SetVerticalFOV(60.f);
}

void Camera::SetFocalLength(const float length)
{
	this->_focalLength = length;
	this->_horizontalFOV = 2.f * atan(0.5f * (float)Scene::GetRenderWidth() / length);
	this->_verticalFOV = 2.f * atan(0.5f * (float)Scene::GetRenderHeight() / length);

	_lastSettedFOV = FOCAL_LENGTH;
}

void Camera::SetHorizontalFOV(const float fov)
{
	this->_horizontalFOV = fov;

	// adjusts vertical fov according to screen ratio
	float zoomX = 1.f / tanf(fov * Deg2Rad() / 2.f);
	float zoomY = zoomX * (float)Scene::GetRenderWidth() / (float)Scene::GetRenderHeight();
	this->_verticalFOV = 2.f * atanf(1.f / zoomY) * Rad2Deg();

	_lastSettedFOV = HORIZONTAL_FOV;
}

void Camera::SetVerticalFOV(const float fov)
{
	this->_verticalFOV = fov;

	// adjusts horizontal fov according to screen ratio
	float zoomY = 1.f / tanf(fov * Deg2Rad() / 2.f);
	float zoomX = zoomY * (float)Scene::GetRenderHeight() / (float)Scene::GetRenderWidth();
	this->_horizontalFOV = 2.f * atanf(1.f / zoomX) * Rad2Deg();

	_lastSettedFOV = VERTICAL_FOV;
}