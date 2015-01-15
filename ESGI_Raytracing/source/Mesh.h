#ifndef MESH_H
#define MESH_H

#include <vector>
#include <limits>
#define __CUDA_INTERNAL_COMPILATION__
#include <math_functions.h>
#undef __CUDA_INTERNAL_COMPILATION__

#include "Vector.h"

struct Face
{
	Face() {}
	Face(int v1Vertex, int v1UV, int v1Normal,
		int v2Vertex, int v2UV, int v2Normal,
		int v3Vertex, int v3UV, int v3Normal)
	{
		v1[0] = v1Vertex; v1[1] = v1UV; v1[2] = v1Normal;
		v2[0] = v2Vertex; v2[1] = v2UV; v2[2] = v2Normal;
		v3[0] = v3Vertex; v3[1] = v3UV; v3[2] = v3Normal;
	}

	int v1[3];
	int v2[3];
	int v3[3];
};

struct AABB
{
	Vector3 minP;
	Vector3 maxP;
};

struct Mesh
{
public:
	__host__ __device__ Mesh() {
		vertexCount = normalCount = uvCount = faceCount = 0;
	}
	//~Mesh()
	//{
	//	this->ClearMesh();
	//}

	Vector3* vertices;
	Vector3* normals;
	Vector2* uvs;
	Face* faces;

	unsigned int vertexCount, normalCount, uvCount, faceCount;

	AABB boundingBox;

	void ClearMesh()
	{
		if (this->vertexCount == 0)
			return;

		delete[] this->vertices;
		delete[] this->normals;
		delete[] this->uvs;
		delete[] this->faces;
	}
	__device__ bool CheckRayIntersection(const Vector3 &rayOrigin, const Vector3 &rayDirection, float &intersectionCoefficient, int &faceIndex)
	{
		if (this->CheckRayAABBIntersection(rayOrigin, rayDirection) == false)
			return false;

		bool faceIntersected = false;

		for (unsigned int i = 0; i < this->faceCount; i++)
		{
			float coeff = intersectionCoefficient;
			faceIntersected = CheckRayFaceIntersection(rayOrigin, rayDirection, coeff, i);
		
			if (coeff < intersectionCoefficient)
			{
				intersectionCoefficient = coeff;
				faceIndex = i;
			}
		}
		
		return faceIntersected;
	}
	__device__ Vector3 GetBarycentricCoord(const Vector3 &intersectionPoint, int faceIndex)
	{
		// Barycentric coordinates for normal and uv interpolation
		// http://stackoverflow.com/questions/17164376/inferring-u-v-for-a-point-in-a-triangle-from-vertex-u-vs
		// http://answers.unity3d.com/questions/383804/calculate-uv-coordinates-of-3d-point-on-plane-of-m.html

		Vector3 p1 = this->vertices[this->faces[faceIndex].v1[0]];
		Vector3 p2 = this->vertices[this->faces[faceIndex].v2[0]];
		Vector3 p3 = this->vertices[this->faces[faceIndex].v3[0]];

		Vector3 f1 = p1 - intersectionPoint;
		Vector3 f2 = p2 - intersectionPoint;
		Vector3 f3 = p3 - intersectionPoint;

		// calculate the areas and factors (order of parameters doesn't matter):
		float a = Vector3::Cross(p1-p2, p1-p3).Magnitude(); // main triangle area a
		float a1 = Vector3::Cross(f2, f3).Magnitude() / a; // p1's triangle area / a
		float a2 = Vector3::Cross(f3, f1).Magnitude() / a; // p2's triangle area / a
		float a3 = Vector3::Cross(f1, f2).Magnitude() / a; // p3's triangle area / a

		return Vector3(a1, a2, a3);
	}
	void GenerateBoundingBox()
	{
		float minX, minY, minZ, maxX, maxY, maxZ;
		minX = minY = minZ = std::numeric_limits<float>::max();
		maxX = maxY = maxZ = std::numeric_limits<float>::min();
		
		for (unsigned int i = 0; i < this->vertexCount; i++)
		{
			if (this->vertices[i].x < minX)
				minX = this->vertices[i].x;
			if (this->vertices[i].y < minY)
				minY = this->vertices[i].y;
			if (this->vertices[i].z < minZ)
				minZ = this->vertices[i].z;
			if (this->vertices[i].x > maxX)
				maxX = this->vertices[i].x;
			if (this->vertices[i].y > maxY)
				maxY = this->vertices[i].y;
			if (this->vertices[i].z > maxZ)
				maxZ = this->vertices[i].z;
		}

		this->boundingBox.minP = Vector3(minX, minY, minZ);
		this->boundingBox.maxP = Vector3(maxX, maxY, maxZ);
	}

private:
	__device__ bool CheckRayAABBIntersection(const Vector3 &rayOrigin, const Vector3 &rayDirection)
	{
		// Based on the answer of "zacharmarz" on gamedev.stackexchange.com:
		// http://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms

		Vector3 dirfrac;
		dirfrac.x = 1.f / rayDirection.x;
		dirfrac.y = 1.f / rayDirection.y;
		dirfrac.z = 1.f / rayDirection.z;

		float t1 = (this->boundingBox.minP.x - rayOrigin.x) * dirfrac.x;
		float t2 = (this->boundingBox.maxP.x - rayOrigin.x) * dirfrac.x;
		float t3 = (this->boundingBox.minP.y - rayOrigin.y) * dirfrac.y;
		float t4 = (this->boundingBox.maxP.y - rayOrigin.y) * dirfrac.y;
		float t5 = (this->boundingBox.minP.z - rayOrigin.z) * dirfrac.z;
		float t6 = (this->boundingBox.maxP.z - rayOrigin.z) * dirfrac.z;

		float tMin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
		float tMax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

		if (tMax < 0.f)
			return false;

		if (tMin > tMax)
			return false;

		return true;
	}
	__device__ bool CheckRayFaceIntersection(const Vector3 &rayOrigin, const Vector3 &rayDirection, float &intersectionCoefficient, int faceIndex)
	{
		// Möller–Trumbore intersection algorithm
		// http://www.scratchapixel.com/lessons/3d-basic-lessons/lesson-9-ray-triangle-intersection/m-ller-trumbore-algorithm/
		float EPSILON = 0.000001f;

		Vector3 e1 = this->vertices[this->faces[faceIndex].v2[0]] - this->vertices[this->faces[faceIndex].v1[0]];
		Vector3 e2 = this->vertices[this->faces[faceIndex].v3[0]] - this->vertices[this->faces[faceIndex].v1[0]];
		Vector3 pvec = Vector3::Cross(rayDirection, e2);
		float det = Vector3::Dot(e1, pvec);
		//if (det < EPSILON) // Forced backface culling
		if (det > -EPSILON && det < EPSILON)
			return false;

		float invDet = 1.f / det;

		Vector3 tvec = rayOrigin - this->vertices[this->faces[faceIndex].v1[0]];
		float u = Vector3::Dot(tvec, pvec) * invDet;
		if (u < 0 || u > 1)
			return false;

		Vector3 qvec = Vector3::Cross(tvec, e1);
		float v = Vector3::Dot(rayDirection, qvec) * invDet;
		if (v < 0 || v > 1 || u + v > 1)
			return false;

		intersectionCoefficient = Vector3::Dot(e2, qvec) * invDet;

		return true;
	}

};

// Left hand operators
inline std::ostream& operator<<(std::ostream& os, const Mesh &mesh)
{

	os << mesh.vertexCount << " Vertices" << std::endl;
	for (unsigned int i = 0; i < mesh.vertexCount; i++)
	{
		os << mesh.vertices[i] << std::endl;
	}

	os << mesh.normalCount << " Normals" << std::endl;
	for (unsigned int i = 0; i < mesh.normalCount; i++)
	{
		os << mesh.normals[i] << std::endl;
	}

	os << mesh.uvCount << " UVs" << std::endl;
	for (unsigned int i = 0; i < mesh.uvCount; i++)
	{
		os << mesh.uvs[i] << std::endl;
	}

	os << mesh.faceCount << " Faces" << std::endl;
	for (unsigned int i = 0; i < mesh.faceCount; i++)
	{
		os << mesh.faces[i].v1[0] << "/" << mesh.faces[i].v1[1] << "/" << mesh.faces[i].v1[2] << " ";
		os << mesh.faces[i].v2[0] << "/" << mesh.faces[i].v2[1] << "/" << mesh.faces[i].v2[2] << " ";
		os << mesh.faces[i].v3[0] << "/" << mesh.faces[i].v3[1] << "/" << mesh.faces[i].v3[2] << std::endl;
	}

	return os;
}

#endif