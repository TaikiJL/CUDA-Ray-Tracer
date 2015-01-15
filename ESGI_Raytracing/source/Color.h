#ifndef COLOR_H
#define COLOR_H

#include <iostream>
#include <algorithm>
#include <device_launch_parameters.h>

#include "MyMath.h"

typedef unsigned char byte;

// Color with RGBA channel from 0.f to 1.f
struct Color
{
	__host__ __device__ Color() : r(0.f), g(0.f), b(0.f), a(1.f) {}
	__host__ __device__ Color(float red, float green, float blue) : r(red), g(green), b(blue), a(1.f) {}
	__host__ __device__ Color(float red, float green, float blue, float alpha) : r(red), g(green), b(blue), a(alpha) {}
	__host__ __device__ Color(float value) : r(value), g(value), b(value), a(1.f) {}
	__host__ __device__ Color(byte red, byte green, byte blue) : r((float)red/255.f), g((float)green/255.f), b((float)blue/255.f), a(1.f) {}
	__host__ __device__ Color(byte red, byte green, byte blue, float alpha) : r((float)red / 255.f), g((float)green / 255.f), b((float)blue / 255.f), a((float)alpha / 255.f) {}

	float r, g, b, a;

	// Right hand operators
	__host__ __device__ Color operator+(const Color &c) const
	{
		return Color(Clampf(this->r + c.r),
			Clampf(this->g + c.g),
			Clampf(this->b + c.b),
			Clampf(this->a + c.a));
	}
	__host__ __device__ Color operator-(const Color &c) const
	{
		return Color(*this + (c * -1.f));
	}
	__host__ __device__ Color operator*(const float a) const
	{
		return Color(Clampf(this->r * a),
			Clampf(this->g * a),
			Clampf(this->b * a),
			Clampf(this->a * a));
	}
	__host__ __device__ Color operator*(const Color c) const
	{
		return Color(Clampf(this->r * c.r),
			Clampf(this->g * c.g),
			Clampf(this->b * c.b),
			Clampf(this->a * c.a));
	}
	__host__ __device__ Color operator/(const float a) const
	{
		return Color(Clampf(this->r / a),
			Clampf(this->g / a),
			Clampf(this->b / a),
			Clampf(this->a / a));
	}
	__host__ __device__ void operator+=(const Color &c) { *this = *this + c; }
	__host__ __device__ void operator-=(const Color &c) { *this = *this - c; }
	__host__ __device__ void operator*=(const float a) { *this = *this * a; }
	__host__ __device__ void operator*=(const Color &c) { *this = *this * c; }
	__host__ __device__ void operator/=(const float a) { *this = *this / a; }
};

inline std::ostream& operator<<(std::ostream& os, const Color &col)
{
	os << "(" << col.r << ", " << col.g << ", " << col.b << ", " << col.a << ")";
    return os;
}

#endif