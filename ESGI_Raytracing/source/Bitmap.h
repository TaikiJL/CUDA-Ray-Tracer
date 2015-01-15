#ifndef BITMAP_H
#define BITMAP_H

#include "Color.h"

enum ImageFormat
{
	BMP,
	JPEG,
	PNG,
	TIFF,
	TARGA
};

class Bitmap
{
public:
	// Constructors
	inline Bitmap(float width, float height) : _width(width), _height(height) { _pixels = new Color[_width*_height]; }
	inline ~Bitmap() { delete[] _pixels; }

	// Getters & Setters
	inline float GetWidth() const { return this->_width; }
	inline float GetHeight() const { return this->_height; }
	void SetPixels(const Color* pixelArray);

	// Static Methods
	void SaveToImage(char* fileName, ImageFormat format);

private:
	int _width, _height;
	Color* _pixels;
};

#endif