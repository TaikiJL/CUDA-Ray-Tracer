#define MAX_THREADS_PER_BLOCK 512

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Scene.h"
#include "Bitmap.h"
#include "Camera.h"
#include "Sphere.h"
#include "Light.h"
#include "MyMath.h"

__device__ int d_width;
__device__ int d_height;
//__device__ Vector3 d_imageRectBottomLeft;
__device__ float d_pixelSize;
//__device__ Camera d_camera;
//__device__ Color d_ambientColor;
__constant__ float d_maxFloat;
__constant__ unsigned int d_maxBounces;

__global__ void RayTracing(Color* pixelArray, int n, Vector3 d_imageRectBottomLeft, Camera d_camera, Color d_ambientColor, Sphere* d_geometries, Light* d_lights)
{
	int i = blockIdx.x * n + threadIdx.x;

	if (threadIdx.x >= n)
		return;

	Vector3 rayDirection = d_imageRectBottomLeft + Vector3(
		((i % d_width) + 0.5f) * d_pixelSize,
		((i / d_width) + 0.5f) * d_pixelSize,
		0.f);
	Vector3 rayOrigin = d_camera.GetPosition();

	const unsigned int maxReflBounce = 1;

	Color pixelStack[maxReflBounce+2];
	float reflCoeff[maxReflBounce+2];
	unsigned int rayTracingIteration = 0;

	int geometryID = -1;

	while (rayTracingIteration <= maxReflBounce)
	{
		float minDistance = d_maxFloat;
		int closestGeometryIndex = 0;

		Vector3 intersectionPoint, intersectionNormal;

		// Find the closest intersection point
		for (int k = 0; k < 3; k++) // HACK
		{
			if (k == geometryID)
				continue;

			if (d_geometries[k].CheckRayIntersection(rayOrigin, rayDirection, intersectionPoint, intersectionNormal))
			{
				float distance = Vector3::Distance(rayOrigin, intersectionPoint);
				if (distance < minDistance)
				{
					minDistance = distance;
					closestGeometryIndex = k;
				}
			}
		}

		// Shading
		if (minDistance >= d_maxFloat)
		{
			pixelStack[rayTracingIteration] = d_camera.GetBackgroundColor();
			reflCoeff[rayTracingIteration] = 0.f;
			break;
		}

		Color diffuse = Color();
		Color specular = Color();
		
		Vector3 incidentVector = Vector3::Normalized(rayOrigin - intersectionPoint);

		for (int k = 0; k < 1; k++) // HACK
		{
			// Lambertian diffuse lighting model
			Vector3 lightVector = Vector3::Normalized(d_lights[k].GetPosition() - intersectionPoint);
			float diff = Vector3::Dot(intersectionNormal, lightVector);
			if (diff < 0.f)
				diff = 0.f;

			diffuse += d_geometries[closestGeometryIndex].GetMaterial().baseColor
				* d_lights[k].GetLightColor()
				* d_lights[k].GetIntensity() * diff;

			if (d_geometries[closestGeometryIndex].GetMaterial().shininess < 0.1f)
				continue;

			// Phong specular lighting model
			Vector3 reflectionVector = 2.f * diff * intersectionNormal - lightVector;

			float dotProduct = Vector3::Dot(incidentVector, reflectionVector);
			if (dotProduct < 0.f)
				dotProduct = 0.f;
			dotProduct = powf(dotProduct, d_geometries[closestGeometryIndex].GetMaterial().shininess * 100.f);

			specular += d_lights[k].GetLightColor() * d_lights[k].GetIntensity()
				* d_geometries[closestGeometryIndex].GetMaterial().specularColor
				* dotProduct;
		}

		pixelStack[rayTracingIteration] = diffuse + specular + (d_ambientColor * d_geometries[closestGeometryIndex].GetMaterial().baseColor);
		reflCoeff[rayTracingIteration] = d_geometries[closestGeometryIndex].GetMaterial().shininess;

		geometryID = closestGeometryIndex;

		float diff = Vector3::Dot(intersectionNormal, incidentVector);

		rayOrigin = intersectionPoint;
		rayDirection = 2.f * diff * intersectionNormal - incidentVector;

		rayTracingIteration++;
	}

	if (rayTracingIteration > 0)
	{
		for (int k = rayTracingIteration - 1; k >= 0; k--)
		{
			pixelStack[k] += pixelStack[k + 1] * reflCoeff[k];
		}
	}

	pixelArray[i] = pixelStack[0];
}

extern "C"
void Render(Color *pixelArray, const Camera &camera)
{
	int width = Scene::GetRenderWidth();
	int height = Scene::GetRenderHeight();
	int totalPixelCount = width * height;
	float maxFloat = std::numeric_limits<float>::max();
	Color ambientColor = Scene::GetAmbientColor();
	unsigned int maxBounces = 2;
	
	// Getting the starting point and a pixel's size
	float imageRectHalfWidth, zoomY;
	zoomY = 1.f / tanf(camera.GetVerticalHalfFOV());
	imageRectHalfWidth = zoomY * tanf(camera.GetHorizontalHalfFOV());

	Vector3 imageRectBottomLeft = Vector3(-imageRectHalfWidth, -1.f, zoomY);
	float pixelSize = 2.f / (float)height;

	// Copy to GPU global memory
	cudaMemcpyToSymbol(d_width, &width, sizeof(int));
	//cudaMemcpyToSymbol(d_imageRectBottomLeft, &imageRectBottomLeft, sizeof(Vector3));
	cudaMemcpyToSymbol(d_pixelSize, &pixelSize, sizeof(float));
	//cudaMemcpyToSymbol(d_camera, &camera, sizeof(Camera));
	//cudaMemcpyToSymbol(d_ambientColor, &ambientColor, sizeof(Color));
	cudaMemcpyToSymbol(d_pixelSize, &pixelSize, sizeof(float));
	cudaMemcpyToSymbol(d_maxFloat, &maxFloat, sizeof(float));
	cudaMemcpyToSymbol(d_maxBounces, &maxBounces, sizeof(unsigned int));

	Sphere* d_geometries;
	Light* d_lights;

	cudaMalloc(&d_geometries, sizeof(Sphere) * Scene::GetGeometries().size());
	cudaMalloc(&d_lights, sizeof(Light)* Scene::GetLights().size());
	cudaMemcpy(d_geometries, &Scene::GetGeometries()[0], sizeof(Sphere) * Scene::GetGeometries().size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_lights, &Scene::GetLights()[0], sizeof(Light)* Scene::GetLights().size(), cudaMemcpyHostToDevice);

	// Device variables declarations
	Color *d_pixelArray;
	
	// Device memory allocations
	cudaMalloc(&d_pixelArray, totalPixelCount * sizeof(Color));

	// Copy pixel array to GPU memory
	cudaMemcpy(d_pixelArray, pixelArray, totalPixelCount * sizeof(Color), cudaMemcpyHostToDevice);
	
	// Invoke kernel here
	int numberOfBlocks = totalPixelCount / MAX_THREADS_PER_BLOCK;
	RayTracing << < numberOfBlocks, MAX_THREADS_PER_BLOCK >> >(d_pixelArray, MAX_THREADS_PER_BLOCK, imageRectBottomLeft, camera, ambientColor, d_geometries, d_lights);
	
	cudaError_t error = cudaDeviceSynchronize();
	//std::cout << "Error: " << cudaGetErrorString(error) << std::endl;

	// Copy pixel array from GPU memory
	cudaMemcpy(pixelArray, d_pixelArray, totalPixelCount * sizeof(Color), cudaMemcpyDeviceToHost);
	
	cudaFree(d_pixelArray);

	//cudaFree(d_geometries);
	//cudaFree(d_lights);
}