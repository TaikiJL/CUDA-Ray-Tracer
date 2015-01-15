#ifndef OBJLOADER_H
#define OBJLOADER_H

#include "Mesh.h"

namespace OBJLoader
{
	bool LoadOBJ(char* path, Mesh &mesh);
}

#endif