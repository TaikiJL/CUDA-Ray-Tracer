#ifndef MATERIAL_H
#define MATERIAL_H

#include "Color.h"

struct Material
{
	Material() {
		this->baseColor = this->specularColor = Color(1.f, 1.f, 1.f);
		this->shininess = 0.5f;
	}

	Color baseColor;
	float shininess;
	Color specularColor; 
};

#endif