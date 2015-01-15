#include "Light.h"

Light::Light()
{
	this->_intensity = 1.f;
	this->_lightType = OMNI;
	this->_color = Color(1.f, 1.f, 1.f);
}