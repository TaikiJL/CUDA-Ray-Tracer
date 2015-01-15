#include "Geometry.h"

#include <limits>

void Geometry::CreateSphereMesh(float radius, int segments)
{
	// create sphere mesh
}

void Geometry::CreateBoxMesh(float length, float width, float height)
{
	this->_mesh.ClearMesh();

	this->_mesh.vertices = new Vector3[8];
	this->_mesh.vertexCount = 8;
	this->_mesh.vertices[0] = Vector3(-width / 2.f, 0.f, -length / 2.f); // bottom left front vertex
	this->_mesh.vertices[1] = Vector3(width / 2.f, 0.f, -length / 2.f); // bottom right front vertex
	this->_mesh.vertices[2] = Vector3(width / 2.f, 0.f, length / 2.f); // bottom right back vertex
	this->_mesh.vertices[3] = Vector3(-width / 2.f, 0.f, length / 2.f); // bottom left back vertex
	this->_mesh.vertices[4] = Vector3(-width / 2.f, height, -length / 2.f); // top left front vertex
	this->_mesh.vertices[5] = Vector3(width / 2.f, height, -length / 2.f); // top right front vertex
	this->_mesh.vertices[6] = Vector3(width / 2.f, height, length / 2.f); // top right back vertex
	this->_mesh.vertices[7] = Vector3(-width / 2.f, height, length / 2.f); // top left back vertex

	this->_mesh.normals = new Vector3[6];
	this->_mesh.normalCount = 6;
	this->_mesh.normals[0] = Vector3(0.f, 0.f, -1.f); // front
	this->_mesh.normals[1] = Vector3(0.f, 0.f, 1.f); // back
	this->_mesh.normals[2] = Vector3(1.f, 0.f, 0.f); // right
	this->_mesh.normals[3] = Vector3(-1.f, 0.f, 0.f); // left
	this->_mesh.normals[4] = Vector3(0.f, 1.f, 0.f); // up
	this->_mesh.normals[5] = Vector3(0.f, -1.f, 0.f); // down

	this->_mesh.uvs = new Vector2[4];
	this->_mesh.uvCount = 4;
	this->_mesh.uvs[0] = Vector2(0.f, 0.f); // bottom left
	this->_mesh.uvs[1] = Vector2(1.f, 0.f); // bottom right
	this->_mesh.uvs[2] = Vector2(1.f, 1.f); // top right
	this->_mesh.uvs[3] = Vector2(0.f, 1.f); // top left

	this->_mesh.faces = new Face[12];
	this->_mesh.faceCount = 12;
	this->_mesh.faces[0] = Face(0, 0, 0, 5, 2, 0, 4, 3, 0); // Front
	this->_mesh.faces[1] = Face(0, 0, 0, 5, 2, 0, 1, 1, 0); // Front
	this->_mesh.faces[2] = Face(3, 1, 1, 6, 3, 1, 2, 0, 1); // Back
	this->_mesh.faces[3] = Face(3, 1, 1, 6, 3, 1, 7, 2, 1); // Back
	this->_mesh.faces[4] = Face(1, 0, 2, 6, 2, 2, 5, 3, 2); // Right
	this->_mesh.faces[5] = Face(1, 0, 2, 6, 2, 2, 2, 1, 2); // Right
	this->_mesh.faces[6] = Face(0, 1, 3, 7, 3, 3, 4, 2, 3); // Left
	this->_mesh.faces[7] = Face(0, 1, 3, 7, 3, 3, 3, 0, 3); // Left
	this->_mesh.faces[8] = Face(4, 0, 4, 6, 3, 4, 5, 2, 4); // Up
	this->_mesh.faces[9] = Face(4, 0, 4, 6, 3, 4, 7, 3, 4); // Up
	this->_mesh.faces[10] = Face(0, 1, 5, 6, 3, 5, 1, 0, 5); // Down
	this->_mesh.faces[11] = Face(0, 1, 5, 6, 3, 5, 3, 2, 5); // Down

	this->_mesh.GenerateBoundingBox();
}

void Geometry::CreatePlaneMesh(float length, float width)
{
	this->_mesh.ClearMesh();

	if (length == 0.f && width == 0.f)
		length = width = 1000.f;

	this->_mesh.vertices = new Vector3[4];
	this->_mesh.vertexCount = 4;
	this->_mesh.vertices[0] = Vector3(-width / 2.f, 0.f, -length / 2.f); // bottom left front vertex
	this->_mesh.vertices[1] = Vector3(width / 2.f, 0.f, -length / 2.f); // bottom right front vertex
	this->_mesh.vertices[2] = Vector3(width / 2.f, 0.f, length / 2.f); // bottom right back vertex
	this->_mesh.vertices[3] = Vector3(-width / 2.f, 0.f, length / 2.f); // bottom left back vertex

	this->_mesh.normals = new Vector3[1];
	this->_mesh.normalCount = 1;
	this->_mesh.normals[0] = Vector3(0.f, 1.f, 0.f); // up

	this->_mesh.uvs = new Vector2[4];
	this->_mesh.uvCount = 4;
	this->_mesh.uvs[0] = Vector2(0.f, 0.f); // bottom left
	this->_mesh.uvs[1] = Vector2(1.f, 0.f); // bottom right
	this->_mesh.uvs[2] = Vector2(1.f, 1.f); // top right
	this->_mesh.uvs[3] = Vector2(0.f, 1.f); // top left

	this->_mesh.faces = new Face[2];
	this->_mesh.faceCount = 2;
	this->_mesh.faces[0] = Face(0, 0, 0, 2, 2, 0, 3, 3, 0);
	this->_mesh.faces[1] = Face(0, 0, 0, 2, 2, 0, 1, 1, 0);

	this->_mesh.GenerateBoundingBox();
}