#ifndef BITMAP_H
#define BITMAP_H

#include "Color.h"

class Bitmap
{
public:
	// Constructors
	Bitmap() { this->_width = this->_height = 0; }
	Bitmap(float width, float height) : _width(width), _height(height) { _pixels = new Color[_width*_height]; }
	Bitmap(char* fileName) { this->LoadTextureImage(fileName); }
	~Bitmap() { if (this->_width != 0 && this->_height != 0) delete[] _pixels; }

	// Getters & Setters
	float GetWidth() const { return this->_width; }
	float GetHeight() const { return this->_height; }
	void SetPixels(const Color* pixelArray);

	// Methods
	void LoadTextureImage(char* fileName);
	void SaveToImage(char* fileName);

private:
	int _width, _height;
	Color* _pixels;
};

#endif