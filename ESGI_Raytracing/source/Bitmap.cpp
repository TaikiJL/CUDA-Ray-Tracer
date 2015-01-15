#include "Bitmap.h"

#include "GL/freeglut.h"
#include "FreeImage.h"

void Bitmap::SetPixels(const Color* pixelArray)
{
	int totalPixelCount = this->_width * this->_height;
	for (int i = 0; i < totalPixelCount; i++)
	{
		this->_pixels[i] = pixelArray[i];
	}
}

void Bitmap::LoadTextureImage(char* fileName)
{
	FIBITMAP* bitmap = NULL;

	FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(fileName);

	bitmap = FreeImage_Load(fif, fileName, JPEG_DEFAULT);

	if (!bitmap)
	{
		std::cout << "Error: image could not be loaded." << std::endl;
		exit(1);
	}

	this->_width = FreeImage_GetWidth(bitmap);
	this->_height = FreeImage_GetHeight(bitmap);
	this->_pixels = new Color[this->_width * this->_height];

	RGBQUAD color;
	int pixelCount = 0;
	for (int i = 0; i < this->_height; i++)
	{
		for (int j = 0; j < this->_width; j++)
		{
			FreeImage_GetPixelColor(bitmap, j, i, &color);
			this->_pixels[pixelCount] = Color((byte)color.rgbRed, (byte)color.rgbGreen, (byte)color.rgbBlue);
						pixelCount++;		}	}
}

void Bitmap::SaveToImage(char* fileName)
{
	FreeImage_Initialise();

	FIBITMAP* bitmap = FreeImage_Allocate(this->_width, this->_height, 24);

	RGBQUAD color;

	if (!bitmap)
	{
		std::cout << "Error: image could not be saved." << std::endl;
		exit(1);
	}

	int pixelCount = 0;
	for (int i = 0; i < this->_height; i++) 
	{
		for (int j = 0; j < this->_width; j++)
		{
			color.rgbRed = this->_pixels[pixelCount].r * 255;
			color.rgbGreen = this->_pixels[pixelCount].g * 255;
			color.rgbBlue = this->_pixels[pixelCount].b * 255;
			FreeImage_SetPixelColor(bitmap, j, i, &color);			pixelCount++;		}	}

	FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(fileName);

	FreeImage_Save(fif, bitmap, fileName, 0);
	
	FreeImage_DeInitialise();
}