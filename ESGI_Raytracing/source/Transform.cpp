#include "Transform.h"

Vector3 Transform::TransformPoint(const Vector3 &v3)
{
	Vector4 v4 = Vector4(v3, 1.f);
	return v3;
}

Vector3 Transform::TransformDirection(const Vector3 &v3)
{
	Vector4 v4 = Vector4(v3, 0.f);
	return v3;
}