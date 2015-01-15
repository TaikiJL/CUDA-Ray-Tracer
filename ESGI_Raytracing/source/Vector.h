#ifndef VECTOR_H
#define VECTOR_H

#include <cmath>
#include <iostream>
#include <device_launch_parameters.h>

struct Vector2
{
	// Constructors
	__host__ __device__ Vector2() : x(0.f), y(0.f) {}
	__host__ __device__ Vector2(float x, float y) : x(x), y(y) {}

	// Right hand operators
	__host__ __device__ Vector2 operator+(const Vector2 &v2) const { return Vector2(this->x + v2.x, this->y + v2.y); }
	__host__ __device__ Vector2 operator-(const Vector2 &v2) const { return Vector2(this->x - v2.x, this->y - v2.y); }
	__host__ __device__ Vector2 operator*(const float a) const { return Vector2(this->x * a, this->y * a); }
	__host__ __device__ Vector2 operator/(const float a) const { return Vector2(this->x / a, this->y / a); }
	__host__ __device__ void operator+=(const Vector2 &v2) { *this = *this + v2; }
	__host__ __device__ void operator-=(const Vector2 &v2) { *this = *this - v2; }
	__host__ __device__ void operator*=(const float a) { *this = *this * a; }
	__host__ __device__ void operator/=(const float a) { *this = *this / a; }

	// Members
	float x, y;
};

// Left hand operators
__host__ __device__ inline Vector2 operator*(float a, const Vector2 &v2) { return v2 * a; }
inline std::ostream& operator<<(std::ostream& os, const Vector2 &v2)
{
	os << "(" << v2.x << ", " << v2.y << ")";
    return os;
}

struct Vector3
{
	// Constructors
	__host__ __device__ Vector3() : x(0.f), y(0.f), z(0.f) {}
	__host__ __device__ Vector3(float x, float y, float z) : x(x), y(y), z(z) {}

	// Methods
	__host__ __device__ float Magnitude() { return sqrt(pow(this->x, 2) + pow(this->y, 2) + pow(this->z, 2)); }
	__host__ __device__ void Normalize() { this->operator/=(this->Magnitude()); }

	// Static Methods
	__host__ __device__ static float Distance(const Vector3 &v3a, const Vector3 &v3b) { return (v3a - v3b).Magnitude(); }
	__host__ __device__ static float Dot(const Vector3 &v3a, const Vector3 &v3b) { return v3a.x * v3b.x + v3a.y * v3b.y + v3a.z * v3b.z; }
	__host__ __device__ static Vector3 Cross(const Vector3 &v3a, const Vector3 &v3b)
	{
		return Vector3(v3a.y * v3b.z - v3a.z * v3b.y,
			v3a.z * v3b.x - v3a.x * v3b.z,
			v3a.x * v3b.y - v3a.y * v3b.x);
	}
	__host__ __device__ static Vector3 Normalized(const Vector3 &v3) { Vector3 normalizedVec = v3; normalizedVec.Normalize(); return normalizedVec; }

	// Right hand operators
	__host__ __device__ Vector3 operator+(const Vector3 &v3) const { return Vector3(this->x + v3.x, this->y + v3.y, this->z + v3.z); }
	__host__ __device__ Vector3 operator-(const Vector3 &v3) const { return Vector3(this->x - v3.x, this->y - v3.y, this->z - v3.z); }
	__host__ __device__ Vector3 operator*(const float a) const { return Vector3(this->x * a, this->y * a, this->z * a); }
	__host__ __device__ Vector3 operator/(const float a) const { return Vector3(this->x / a, this->y / a, this->z / a); }
	__host__ __device__ void operator+=(const Vector3 &v3) { *this = *this + v3; }
	__host__ __device__ void operator-=(const Vector3 &v3) { *this = *this - v3; }
	__host__ __device__ void operator*=(const float a) { *this = *this * a; }
	__host__ __device__ void operator/=(const float a) { *this = *this / a; }

	// Members
	float x, y, z;
};

// Left hand operators
inline __host__ __device__ Vector3 operator*(float a, const Vector3 &v3) { return v3 * a; }
inline std::ostream& operator<<(std::ostream& os, const Vector3 &v3)
{
	os << "(" << v3.x << ", " << v3.y << ", " << v3.z << ")";
    return os;
}

struct Vector4
{
	// Constructors
	__host__ __device__ Vector4() : x(0.f), y(0.f), z(0.f), w(0.f) {}
	__host__ __device__ Vector4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
	__host__ __device__ Vector4(Vector3 v3, float w)
	{ 
		this->x = v3.x;
		this->y = v3.y;
		this->z = v3.z;
		this->w = w;
	}

	// Static Functions
	__host__ __device__ static float Dot(const Vector4 &v4a, const Vector4 &v4b) { return v4a.x * v4b.x + v4a.y * v4b.y + v4a.z * v4b.z + v4a.w * v4b.w; }

	// Right hand operators
	__host__ __device__ Vector4 operator+(const Vector4 &v4) const
	{
		return Vector4(this->x + v4.x,
			this->y + v4.y,
			this->z + v4.z,
			this->w + v4.w);
	}
	__host__ __device__ Vector4 operator-(const Vector4 &v4) const
	{
		return Vector4(this->x - v4.x,
			this->y - v4.y,
			this->z - v4.z,
			this->w - v4.w);
	}
	__host__ __device__ Vector4 operator*(const float a) const
	{
		return Vector4(this->x * a,
			this->y * a,
			this->z * a,
			this->w * a);
	}
	__host__ __device__ Vector4 operator/(const float a) const
	{
		return Vector4(this->x / a,
			this->y / a,
			this->z / a,
			this->w / a);
	}
	__host__ __device__ void operator+=(const Vector4 &v4) { *this = *this + v4; }
	__host__ __device__ void operator-=(const Vector4 &v4) { *this = *this - v4; }
	__host__ __device__ void operator*=(const float a) { *this = *this * a; }
	__host__ __device__ void operator/=(const float a) { *this = *this / a; }

	// Members
	float x, y, z, w;
};

// Left hand operators
inline Vector4 operator*(float a, const Vector4 &v4) { return v4 * a; }
inline std::ostream& operator<<(std::ostream& os, const Vector4 &v4)
{
	os << "(" << v4.x << ", " << v4.y << ", " << v4.z << ", " << v4.w << ")";
    return os;
}

#endif