#ifndef MYMATH_H
#define MYMATH_H

#define M_PI 3.141592654f
#define M_PI_2 1.570796327f

#include <device_launch_parameters.h>
#include <math_constants.h>

__host__ __device__ static float Deg2Rad()
{
	return M_PI_2 / 360.f;
}

__host__ __device__ static float Rad2Deg()
{
	return 360.f / M_PI_2;
}

template <typename T>
__host__ __device__ static T Clamp(T value, T min, T max)
{
	T result;
	if (value > max)
		result = max;
	else if (value < min)
		result = min;
	else
		result = value;

	return result;
}

__host__ __device__ static float Clampf(float value, float min = 0.f, float max = 1.f)
{
	return Clamp<float>(value, min, max);
}

#endif