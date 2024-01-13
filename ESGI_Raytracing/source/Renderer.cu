#define MAX_THREADS_PER_BLOCK 512

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

#include "Scene.h"
#include "Bitmap.h"
#include "Camera.h"
#include "Geometry.h"
#include "Light.h"
#include "MyMath.h"

__device__ int s_width;
__device__ int s_height;
__device__ float s_pixelSize;
__device__ Geometry* s_geometries;
__device__ Light* s_lights;
__device__ int s_geometryCount;
__device__ int s_lightCount;
__device__ unsigned short int s_maxBounce;

__global__ void RayTracing(Color *pixelArray, int n, Vector3 d_imageRectBottomLeft, Camera d_camera, Color d_ambientColor, Geometry* d_geometries, Light* d_lights)
{
	int i = blockIdx.x * n + threadIdx.x;
	
	if (threadIdx.x >= n)
		return;
	
	int x = i % s_width;
	int y = i / s_width;
	
	Vector3 rayDirection = d_imageRectBottomLeft + Vector3((x + 0.5f) * s_pixelSize, (y + 0.5f) * s_pixelSize, 0.f);
	rayDirection = d_camera.TransformDirection(rayDirection);
	Vector3 rayOrigin = d_camera.GetPosition();
	float intersectionCoefficient = d_camera.GetFarClip();
	int closestGeometryIndex = 0;
	int faceIndex = 0;

	for (int k = 0; k < s_geometryCount; k++)
	{
		float coeff = d_camera.GetFarClip();
		
		d_geometries[k].CheckRayIntersection(rayOrigin, rayDirection, coeff, faceIndex);
	
		if (coeff < intersectionCoefficient)
		{
			intersectionCoefficient = coeff;
			closestGeometryIndex = k;
		}
	}

	// Shading
	if (intersectionCoefficient >= d_camera.GetFarClip())
	{
		pixelArray[i] = d_camera.GetBackgroundColor();
		return;
	}
	
	Vector3 intersectionPoint = (rayDirection - rayOrigin) * intersectionCoefficient;
	intersectionPoint += rayOrigin;

	Geometry* geometry = &d_geometries[closestGeometryIndex];

	Vector3 barycentricCoord = geometry->GetMesh().GetBarycentricCoord(
		geometry->InverseTransformPoint(intersectionPoint), faceIndex);
	
	Vector3* geometryNormals = geometry->GetMesh().normals;
	Face* geometryFaces = geometry->GetMesh().faces;
	Vector2* geometryUVs = geometry->GetMesh().uvs;

	// find the normal corresponding to intersection:
	Vector3 intersectionNormal = barycentricCoord.x * geometryNormals[geometryFaces[faceIndex].v1[2]] +
			barycentricCoord.y * geometryNormals[geometryFaces[faceIndex].v2[2]] +
			barycentricCoord.z * geometryNormals[geometryFaces[faceIndex].v3[2]];

	intersectionNormal.Normalize();
	intersectionNormal = geometry->TransformDirection(intersectionNormal);

	// find the uv corresponding to intersection:
	Vector2 uv = barycentricCoord.x * geometryUVs[geometryFaces[faceIndex].v1[1]] +
			barycentricCoord.y * geometryUVs[geometryFaces[faceIndex].v2[1]] +
			barycentricCoord.z * geometryUVs[geometryFaces[faceIndex].v3[1]];

	Color albedo = Color();
	Color specular = Color();
	
	for (int k = 0; k < s_lightCount; k++)
	{
		// Lambertian diffuse lighting model
		Vector3 lightVector = Vector3::Normalized(d_lights[k].GetPosition() - intersectionPoint);
		float dotProduct = Vector3::Dot(intersectionNormal, lightVector);
		if (dotProduct < 0.0f)
			dotProduct = 0.f;
		albedo += geometry->GetMaterial().baseColor
			* d_lights[k].GetLightColor()
			* d_lights[k].GetIntensity() * dotProduct;
	
		// Phong specular lighting model
		Vector3 reflectionVector = 2.f * dotProduct * intersectionNormal - lightVector;
		Vector3 viewVector = Vector3::Normalized(d_camera.GetPosition() - intersectionPoint);
		dotProduct = Vector3::Dot(viewVector, reflectionVector);
		if (dotProduct < 0.0f)
			dotProduct = 0.0f;
		dotProduct = powf(dotProduct, geometry->GetMaterial().shininess * 100.f);
		
		specular += d_lights[k].GetLightColor() * d_lights[k].GetIntensity()
			* geometry->GetMaterial().specularColor
			* dotProduct;
	}
	
	pixelArray[i] = albedo + specular + d_ambientColor * geometry->GetMaterial().baseColor;
}

__global__ void RayTracingIsometric(Color *pixelArray, int n, Vector3 d_imageRectBottomLeft, Camera d_camera, Color d_ambientColor, Geometry* d_geometries, Light* d_lights)
{
	int i = blockIdx.x * n + threadIdx.x;

	if (threadIdx.x >= n)
		return;

	int x = i % s_width;
	int y = i / s_width;

	Vector3 rayDirection = d_camera.TransformDirection(Vector3(0.f, 0.f, 1.f));
	Vector3 rayOrigin = d_imageRectBottomLeft + Vector3((x + 0.5f) * s_pixelSize, (y + 0.5f) * s_pixelSize, 0.f);
	rayOrigin = d_camera.TransformPoint(rayOrigin);
	float intersectionCoefficient = d_camera.GetFarClip();
	int closestGeometryIndex = 0;
	int faceIndex = 0;

	for (int k = 0; k < s_geometryCount; k++)
	{
		float coeff = d_camera.GetFarClip();

		d_geometries[k].CheckRayIntersection(rayOrigin, rayDirection, coeff, faceIndex);

		if (coeff < intersectionCoefficient)
		{
			intersectionCoefficient = coeff;
			closestGeometryIndex = k;
		}
	}

	// Shading
	if (intersectionCoefficient >= d_camera.GetFarClip())
	{
		pixelArray[i] = d_camera.GetBackgroundColor();
		return;
	}

	Vector3 intersectionPoint = (rayDirection - rayOrigin) * intersectionCoefficient;
	intersectionPoint += rayOrigin;

	Geometry* geometry = &d_geometries[closestGeometryIndex];

	Vector3 barycentricCoord = geometry->GetMesh().GetBarycentricCoord(
		geometry->InverseTransformPoint(intersectionPoint), faceIndex);
	
	Vector3* geometryNormals = geometry->GetMesh().normals;
	Face* geometryFaces = geometry->GetMesh().faces;

	// find the normal corresponding to point f:
	Vector3 intersectionNormal = barycentricCoord.x * geometryNormals[geometryFaces[faceIndex].v1[2]] +
			barycentricCoord.y * geometryNormals[geometryFaces[faceIndex].v2[2]] +
			barycentricCoord.z * geometryNormals[geometryFaces[faceIndex].v3[2]];

	intersectionNormal.Normalize();
	intersectionNormal = geometry->TransformDirection(intersectionNormal);

	Color albedo = Color();
	Color specular = Color();
	
	for (int k = 0; k < s_lightCount; k++)
	{
		// Lambertian diffuse lighting model
		Vector3 lightVector = Vector3::Normalized(d_lights[k].GetPosition() - intersectionPoint);
		float dotProduct = Vector3::Dot(intersectionNormal, lightVector);
		if (dotProduct < 0.0f)
			dotProduct = 0.f;
		albedo += geometry->GetMaterial().baseColor
			* d_lights[k].GetLightColor()
			* d_lights[k].GetIntensity() * dotProduct;
	
		// Phong specular lighting model
		Vector3 reflectionVector = 2.f * dotProduct * intersectionNormal - lightVector;
		Vector3 viewVector = Vector3::Normalized(d_camera.GetPosition() - intersectionPoint);
		dotProduct = Vector3::Dot(viewVector, reflectionVector);
		if (dotProduct < 0.0f)
			dotProduct = 0.0f;
		dotProduct = powf(dotProduct, geometry->GetMaterial().shininess * 100.f);
		
		specular += d_lights[k].GetLightColor() * d_lights[k].GetIntensity()
			* geometry->GetMaterial().specularColor
			* dotProduct;
	}
	
	pixelArray[i] = albedo + specular + d_ambientColor * geometry->GetMaterial().baseColor;
}

extern "C"
void Render(Color *pixelArray, const Camera &camera)
{
	int width = Scene::GetRenderWidth();
	int height = Scene::GetRenderHeight();
	int totalPixelCount = width * height;
	Color ambientColor = Scene::GetAmbientColor();
	int geometryCount = Scene::GetGeometries().size();
	int lightCount = Scene::GetLights().size();
	unsigned short int maxBounce = Scene::GetMaxBounce();

	float pixelSize;
	if (camera.GetCameraType() == PERSPECTIVE)
	{
		pixelSize = 2.f / (float)height;
	}
	else if (camera.GetCameraType() == ORTHOGRAPHIC)
	{
		pixelSize = camera.GetSize() / (float)height;
	}

	// Copy to GPU global memory
	cudaMemcpyToSymbol(s_width, &width, sizeof(int));
	cudaMemcpyToSymbol(s_pixelSize, &pixelSize, sizeof(float));
	cudaMemcpyToSymbol(s_geometryCount, &geometryCount, sizeof(int));
	cudaMemcpyToSymbol(s_lightCount, &lightCount, sizeof(int));
	cudaMemcpyToSymbol(s_maxBounce, &maxBounce, sizeof(unsigned short int));

	// Copy pointers to GPU global memory
	Geometry* d_geometries;
	cudaGetSymbolAddress((void**)&d_geometries, s_geometries);
	cudaMalloc(&d_geometries, sizeof(Geometry) * geometryCount);
	cudaMemcpy(d_geometries, &Scene::GetGeometries()[0], sizeof(Geometry) * geometryCount, cudaMemcpyHostToDevice);

	Light* d_lights;
	cudaGetSymbolAddress((void**)&d_lights, s_lights);
	cudaMalloc(&d_lights, sizeof(Light) * lightCount);
	cudaMemcpy(d_lights, &Scene::GetLights()[0], sizeof(Light) * lightCount, cudaMemcpyHostToDevice);

	// Geometries' meshes deep copy
	Mesh* d_meshes = new Mesh[Scene::GetGeometries().size()];
	for (unsigned int i = 0; i < Scene::GetGeometries().size(); i++)
	{
		Mesh* mesh = &Scene::GetGeometries()[i].GetMesh();

		cudaMalloc(&(d_meshes[i].vertices), mesh->vertexCount * sizeof(Vector3));
		cudaMalloc(&(d_meshes[i].normals), mesh->normalCount * sizeof(Vector3));
		cudaMalloc(&(d_meshes[i].uvs), mesh->uvCount * sizeof(Vector2));
		cudaMalloc(&(d_meshes[i].faces), mesh->faceCount * sizeof(Face));

		cudaMemcpy(d_meshes[i].vertices, mesh->vertices, mesh->vertexCount * sizeof(Vector3), cudaMemcpyHostToDevice);
		cudaMemcpy(d_meshes[i].normals, mesh->normals, mesh->normalCount * sizeof(Vector3), cudaMemcpyHostToDevice);
		cudaMemcpy(d_meshes[i].uvs, mesh->uvs, mesh->uvCount * sizeof(Vector2), cudaMemcpyHostToDevice);
		cudaMemcpy(d_meshes[i].faces, mesh->faces, mesh->faceCount * sizeof(Face), cudaMemcpyHostToDevice);

		cudaMemcpy(&(d_geometries[i].GetMesh().vertices), &(d_meshes[i].vertices), sizeof(Vector3*), cudaMemcpyHostToDevice);
		cudaMemcpy(&(d_geometries[i].GetMesh().normals), &(d_meshes[i].normals), sizeof(Vector3*), cudaMemcpyHostToDevice);
		cudaMemcpy(&(d_geometries[i].GetMesh().uvs), &(d_meshes[i].uvs), sizeof(Vector2*), cudaMemcpyHostToDevice);
		cudaMemcpy(&(d_geometries[i].GetMesh().faces), &(d_meshes[i].faces), sizeof(Face*), cudaMemcpyHostToDevice);
	}

	// Copy pixel array to GPU memory
	Color *d_pixelArray;
	cudaMalloc(&d_pixelArray, totalPixelCount * sizeof(Color));
	cudaMemcpy(d_pixelArray, pixelArray, totalPixelCount * sizeof(Color), cudaMemcpyHostToDevice);
	
	// Invoke kernel here
	int numberOfBlocks = totalPixelCount / MAX_THREADS_PER_BLOCK;
	if (camera.GetCameraType() == PERSPECTIVE)
	{
		// Getting the starting point and a pixel's size
		float imageRectHalfWidth, zoomY;
		zoomY = 1.f / tanf(camera.GetVerticalHalfFOV());
		imageRectHalfWidth = zoomY * tanf(camera.GetHorizontalHalfFOV());
		Vector3 imageRectBottomLeft = Vector3(-imageRectHalfWidth, -1.f, zoomY);

		RayTracing<<<numberOfBlocks, MAX_THREADS_PER_BLOCK>>>(d_pixelArray, MAX_THREADS_PER_BLOCK, imageRectBottomLeft, camera, ambientColor, d_geometries, d_lights);
	}
	else if (camera.GetCameraType() == ORTHOGRAPHIC)
	{
		Vector3 imageRectBottomLeft = Vector3((-(float)width * camera.GetSize()) / ((float)height * 2.f), -camera.GetSize() / 2.f, 0.f);
		RayTracingIsometric<<<numberOfBlocks, MAX_THREADS_PER_BLOCK>>>(d_pixelArray, MAX_THREADS_PER_BLOCK, imageRectBottomLeft, camera, ambientColor, d_geometries, d_lights);
	}
	
	cudaError_t error = cudaDeviceSynchronize();
	//std::cout << "Error Render: " << cudaGetErrorString(error) << std::endl;

	// Copy pixel array from GPU memory
	cudaMemcpy(pixelArray, d_pixelArray, totalPixelCount * sizeof(Color), cudaMemcpyDeviceToHost);
	
	// Free memories
	cudaFree(d_pixelArray);

	cudaFree(d_geometries);
	cudaFree(d_lights);

	for (unsigned int i = 0; i < Scene::GetGeometries().size(); i++)
	{
		cudaFree(d_meshes[i].vertices);
		cudaFree(d_meshes[i].normals);
		cudaFree(d_meshes[i].uvs);
		cudaFree(d_meshes[i].faces);
	}

	delete[] d_meshes;
}