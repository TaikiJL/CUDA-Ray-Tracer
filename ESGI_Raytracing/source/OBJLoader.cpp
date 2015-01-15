#include <string.h>
#include <vector>

#include "OBJLoader.h"
#include "Mesh.h"

static const unsigned char MAX_CHAR_SIZE = 64;

namespace OBJLoader
{
	bool LoadOBJ(char* path, Mesh &mesh)
	{
		FILE* file;
		fopen_s(&file, path, "r");
		if (file == NULL)
		{
			printf("Opening file failed");
			return false;
		}
		
		std::vector<Vector3> vertices, normals;
		std::vector<Vector2> uvs;
		std::vector<Face> faces;

		while (true)
		{
			char lineHeader[MAX_CHAR_SIZE];
			
			// read the first word of the line
			int res = fscanf_s(file, "%s", lineHeader, sizeof(lineHeader));
			if (res == EOF)
				break;

			if (strcmp(lineHeader, "v") == 0)
			{
				Vector3 vertex;
				fscanf_s(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
				vertices.push_back(vertex);
			}
			else if (strcmp(lineHeader, "vt") == 0)
			{
				Vector2 uv;
				fscanf_s(file, "%f %f\n", &uv.x, &uv.y);
				uvs.push_back(uv);
			}
			else if (strcmp(lineHeader, "vn") == 0)
			{
				Vector3 normal;
				fscanf_s(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
				normals.push_back(normal);
			}
			else if (strcmp(lineHeader, "f") == 0)
			{
				Face face;
				int matches = fscanf_s(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n",
					&face.v1[0], &face.v1[1], &face.v1[2],
					&face.v2[0], &face.v2[1], &face.v2[2],
					&face.v3[0], &face.v3[1], &face.v3[2]);
				if (matches != 9)
				{
					printf("File can't be read by our simple parser : ( Try exporting with other options\n");
					return false;
				}

				// Indexes for .obj faces start at 1. Our array's start at 0.
				for (int i = 0; i < 3; i++)
				{
					face.v1[i]--;
					face.v2[i]--;
					face.v3[i]--;
				}

				faces.push_back(face);
			}
		}

		mesh.ClearMesh();
		
		mesh.vertexCount = vertices.size();
		mesh.normalCount = normals.size();
		mesh.uvCount = uvs.size();
		mesh.faceCount = faces.size();

		mesh.vertices = new Vector3[vertices.size()];
		mesh.normals = new Vector3[normals.size()];
		mesh.uvs = new Vector2[uvs.size()];
		mesh.faces = new Face[faces.size()];

		for (unsigned int i = 0; i < vertices.size(); i++)
		{
			mesh.vertices[i] = vertices[i];
		}
		for (unsigned int i = 0; i < normals.size(); i++)
		{
			mesh.normals[i] = normals[i];
		}
		for (unsigned int i = 0; i < uvs.size(); i++)
		{
			mesh.uvs[i] = uvs[i];
		}
		for (unsigned int i = 0; i < faces.size(); i++)
		{
			mesh.faces[i] = faces[i];
		}

		mesh.GenerateBoundingBox();

		return true;
	}
}