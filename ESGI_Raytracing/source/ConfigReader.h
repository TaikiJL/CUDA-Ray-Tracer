#ifndef CONFIG_READER_H
#define CONFIG_READER_H

#include <vector>

class Camera;
class Light;
class Geometry;

struct Block;
struct ParameterBlock;
struct TransformInformation;

enum ParameterType
{
	_NONE,
	_FLOAT,
	_INTEGER,
	_STRING
};

enum ConfigObjectType
{
	_UNKNOWN_OBJECT,
	_SPHERE,
	_BOX,
	_INF_CYLINDER,
	_INF_CONE,
	_DISC,
	_TRIANGLE,
	_SQUARE,
	_PLANE,
	_OBJ
};

enum ConfigLightType
{
	_UNKNOWN_LIGHT,
	_POINT,
	_SPOT,
	_DIRECTIONAL
};

enum ConfigCameraType
{
	_UNKNOWN_CAMERA,
	_ORTHOGRAPHIC,
	_PERSPECTIVE,
	_PANORAMIC
};

class ConfigReader
{
private:
	char* filePath;

public:

	ConfigReader(char* path);
	~ConfigReader();

	bool readConfig(std::vector<Light>* lights, std::vector<Camera>* cameras, std::vector<Geometry>* objects);

	bool handleToken(std::string token, std::vector<Block>* blocks);

	void processBlocks(std::vector<Block>* blocks, std::vector<Light>* lights, std::vector<Camera>* cameras, std::vector<Geometry>* objects);

	ParameterBlock searchForParameterInBlock(Block* block, std::string paramName, ParameterType param1, ParameterType param2 = _NONE, ParameterType param3 = _NONE);

	TransformInformation getTransformInformation(Block* block);

	ConfigObjectType getObjectType(Block* block);

	ConfigLightType getLightType(Block* block);

	ConfigCameraType getCameraType(Block* block);

	void assignDefaultParameterValue(ParameterBlock* params, std::string paramName);

};

#endif