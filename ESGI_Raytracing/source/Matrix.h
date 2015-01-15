#ifndef MATRIX_H
#define MATRIX_H

#include "Vector.h"

struct Matrix4x3
{
	// Constructors
	__host__ __device__ Matrix4x3() {}
	__host__ __device__ Matrix4x3(float m11, float m12, float m13, float m14,
		float m21, float m22, float m23, float m24,
		float m31, float m32, float m33, float m34)
	{
		this->rows[0] = Vector4(m11, m12, m13, m14);
		this->rows[1] = Vector4(m21, m22, m23, m24);
		this->rows[2] = Vector4(m31, m32, m33, m34);
	}

	// Methods
	__host__ __device__ Vector3 Multiply(const Vector4& v4) const
	{
		return Vector3(Vector4::Dot(this->rows[0], v4),
			Vector4::Dot(this->rows[1], v4),
			Vector4::Dot(this->rows[2], v4));
	}

	// Right hand operators
	__host__ __device__ Vector3 operator*(const Vector4& v4) const { return this->Multiply(v4); }

	// Members
	Vector4 rows[3];
};

struct Matrix4x4
{
	// Constructors
	__host__ __device__ Matrix4x4() {}
	__host__ __device__ Matrix4x4(float m11, float m12, float m13, float m14,
		float m21, float m22, float m23, float m24,
		float m31, float m32, float m33, float m34,
		float m41, float m42, float m43, float m44)
	{
		this->rows[0] = Vector4(m11, m12, m13, m14);
		this->rows[1] = Vector4(m21, m22, m23, m24);
		this->rows[2] = Vector4(m31, m32, m33, m34);
		this->rows[3] = Vector4(m41, m42, m43, m44);
	}

	// Methods
	__host__ __device__ Vector4 Multiply (const Vector4& v4) const
	{
		return Vector4(Vector4::Dot(this->rows[0], v4),
			Vector4::Dot(this->rows[1], v4),
			Vector4::Dot(this->rows[2], v4),
			Vector4::Dot(this->rows[3], v4));
	}
	__host__ __device__ Matrix4x4 Multiply(const Matrix4x4& m44) const
	{
		Vector4 columns[4];
		columns[0] = this->Multiply(Vector4(m44.rows[0].x, m44.rows[1].x, m44.rows[2].x, m44.rows[3].x));
		columns[1] = this->Multiply(Vector4(m44.rows[0].y, m44.rows[1].y, m44.rows[2].y, m44.rows[3].y));
		columns[2] = this->Multiply(Vector4(m44.rows[0].z, m44.rows[1].z, m44.rows[2].z, m44.rows[3].z));
		columns[3] = this->Multiply(Vector4(m44.rows[0].w, m44.rows[1].w, m44.rows[2].w, m44.rows[3].w));

		return Matrix4x4(columns[0].x, columns[1].x, columns[2].x, columns[3].x,
			columns[0].y, columns[1].y, columns[2].y, columns[3].y,
			columns[0].z, columns[1].z, columns[2].z, columns[3].z,
			columns[0].w, columns[1].w, columns[2].w, columns[3].w);
	}

	// Right hand operators
	__host__ __device__ Vector4 operator*(const Vector4& v4) const { return this->Multiply(v4); }
	__host__ __device__ Matrix4x4 operator*(const Matrix4x4& m44) const { return this->Multiply(m44); }

	// Members
	Vector4 rows[4];
};

#endif