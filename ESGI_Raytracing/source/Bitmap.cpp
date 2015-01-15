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

void Bitmap::SaveToImage(char* fileName, ImageFormat format)
{
	FreeImage_Initialise();

	FIBITMAP* bitmap = FreeImage_Allocate(this->_width, this->_height, 24);

	RGBQUAD color;

	if (!bitmap)
		exit(1); //WTF?! We can't even allocate images? Die!

	int pixelCount = 0;
	for (int i = 0; i < this->_height; i++) 
	{
		for (int j = 0; j < this->_width; j++)
		{
			color.rgbRed = this->_pixels[pixelCount].r * 255;
			color.rgbGreen = this->_pixels[pixelCount].g * 255;
			color.rgbBlue = this->_pixels[pixelCount].b * 255;
			FreeImage_SetPixelColor(bitmap, j, i, &color);			pixelCount++;		}	}

	FREE_IMAGE_FORMAT fif;
	switch (format)
	{
	case BMP:
		fif = FIF_BMP;
		break;
	case JPEG:
		fif = FIF_JPEG;
		break;
	case PNG:
		fif = FIF_PNG;
		break;
	case TIFF:
		fif = FIF_TIFF;
		break;
	case TARGA:
		fif = FIF_TARGA;
		break;
	default:
		break;
	}

	FreeImage_Save(fif, bitmap, fileName, 0);
	
	FreeImage_DeInitialise();
}