//////////////////////////////////
/*
 *
 *
 */
//////////////////////////////////

#include <iostream>
#include <fstream>
#include <string>

#include "ConfigReader.h"
#include "Camera.h"
#include "Light.h"
#include "Geometry.h"
#include "Scene.h"
#include "OBJLoader.h"
#include "Bitmap.h"
#include "PathUtils.h"

static const char DELIMITING_CHARACTER = ' ';

struct Block
{
	std::vector<std::string> tokens;
};

ConfigReader::ConfigReader(char* path)
{
	filePath = path;
}

ConfigReader::~ConfigReader()
{

}

bool ConfigReader::readConfig(std::vector<Light>* lights, std::vector<Camera>* cameras, std::vector<Geometry>* objects)
{
	std::string line;
	std::ifstream configFile;
	configFile.open(filePath);
	//std::cout << "Reading \"" << filePath << "\" \n";

	if(!configFile.is_open())
	{
		std::cout << "ERROR Reading Config File\n";
		return false;
	}

	std::vector<Block> blocks;

	while(!configFile.eof())
	{
		getline(configFile, line);

		if(line.length() == 0 || line[0] == '#')
			continue;

		size_t pos = std::min(line.find(DELIMITING_CHARACTER), line.length());
		std::string token;
		bool moveToNextLine = false;

		while (!moveToNextLine)
		{
			token = line.substr(0, pos);
			handleToken(token, &blocks);
			line.erase(0, pos + 1);
			pos = std::min(line.find(DELIMITING_CHARACTER), line.length());
			if(pos == std::string::npos || line.length() == 0)
			{
				if(line.length() != 0)
				{
					token = line.substr(0, line.length());
					handleToken(token, &blocks);
				}
				moveToNextLine = true;
			}
		}
	}
	configFile.close();

	processBlocks(&blocks, lights, cameras, objects);

	//std::cout << "SUCCESS Reading Config File\n";
	//system("pause");
	return true;
}

bool ConfigReader::handleToken(std::string token, std::vector<Block>* blocks)
{
	if(token == "model" || token == "camera" || token == "light" || token == "object")
	{
		Block newBlock;
		newBlock.tokens.push_back(token);
		blocks->push_back(newBlock);
	} else if(token == "end")
	{

	}
	else
	{
		(*blocks).back().tokens.push_back(token);
	}

	return true;
}

struct TransformInformation
{
	Vector3 _rotate;
	Vector3 _translate;
	Vector3 _scale;
};

struct ParameterBlock
{
	int intParameters[3];
	float floatParameters[3];
	std::string stringParameters[3];
};

void ConfigReader::processBlocks(std::vector<Block>* blocks, std::vector<Light>* lights, std::vector<Camera>* cameras, std::vector<Geometry>* objects)
{

	for(int i = 0; i < blocks->size(); i++)
	{
		Block block = (*blocks)[i];

		// print block content
		//for(int j = 0; j < block.tokens.size(); j++)
		//{
		//	std::cout << block.tokens[j] << "\t";
		//}
		//std::cout << std::endl;

		if(block.tokens[0] == "model")
		{
			ParameterBlock ambient = searchForParameterInBlock(&block, "ambient", _INTEGER, _INTEGER, _INTEGER);
			int depth		= searchForParameterInBlock(&block, "depth", _INTEGER).intParameters[0];
			float threshold = searchForParameterInBlock(&block, "threshold", _FLOAT).floatParameters[0];

			ParameterBlock textureInfo = searchForParameterInBlock(&block, "texture", _STRING);

			std::cout << "Read Model Block \n";
			
			Scene::SetAmbientColor(Color((byte)ambient.intParameters[0],
				(byte)ambient.intParameters[1],
				(byte)ambient.intParameters[2]));

			Scene::SetMaxBounce((unsigned short int)depth);
		}
		else if(block.tokens[0] == "camera")
		{
			ConfigCameraType cameraType = getCameraType(&block);

			float cameraFloatParameter = 0;
			if(cameraType == _PERSPECTIVE)
				cameraFloatParameter = searchForParameterInBlock(&block, "perspective", _FLOAT).floatParameters[0];
			else if (cameraType == _ORTHOGRAPHIC)
				cameraFloatParameter = searchForParameterInBlock(&block, "isometric", _FLOAT).floatParameters[0];

			TransformInformation transInfo = getTransformInformation(&block);

			float width	 = searchForParameterInBlock(&block, "width", _INTEGER).intParameters[0];
			float height = searchForParameterInBlock(&block, "height", _INTEGER).intParameters[0];

			ParameterBlock colorInfo = searchForParameterInBlock(&block, "background_color", _INTEGER, _INTEGER, _INTEGER);

			std::cout << "Read Camera Block \n";

			Scene::SetRenderResolution(width, height);

			Camera cam = Camera();

			if (cameraType == _PERSPECTIVE)
				cam.SetFocalLength(cameraFloatParameter);
			else if (cameraType == _ORTHOGRAPHIC)
				cam.SetCameraType(ORTHOGRAPHIC, cameraFloatParameter);
			else if (cameraType == _PANORAMIC)
				cam.SetCameraType(PANORAMIC);

			cam.SetBackgroundColor(Color((byte)colorInfo.intParameters[0],
				(byte)colorInfo.intParameters[1],
				(byte)colorInfo.intParameters[2]));

			cam.SetPosition(transInfo._translate);
			cam.SetRotation(transInfo._rotate);
			cam.SetScale(transInfo._scale);

			cameras->push_back(cam);
		}
		else if(block.tokens[0] == "light")
		{
			ConfigLightType lightType = getLightType(&block);

			float spotAngle = 0;
			if(lightType == _SPOT)
				spotAngle = searchForParameterInBlock(&block, "spot", _FLOAT).floatParameters[0];

			TransformInformation transInfo = getTransformInformation(&block);

			float intensity = searchForParameterInBlock(&block, "intensity", _FLOAT).floatParameters[0];

			ParameterBlock colorInfo = searchForParameterInBlock(&block, "color", _INTEGER, _INTEGER, _INTEGER);

			std::cout << "Read Light Block \n";

			Light light = Light();
			light.SetPosition(transInfo._translate);
			light.SetRotation(transInfo._rotate);
			light.SetScale(transInfo._scale);

			light.SetLightColor(Color((byte)colorInfo.intParameters[0],
				(byte)colorInfo.intParameters[1],
				(byte)colorInfo.intParameters[2]));

			light.SetIntensity(intensity);

			lights->push_back(light);
		}
		else if(block.tokens[0] == "object")
		{
			ConfigObjectType objType = getObjectType(&block);

			std::string objPath;

			ParameterBlock primitiveInfo;

			switch (objType)
			{
			case _UNKNOWN_OBJECT:
				break;
			case _SPHERE:
				break;
			case _BOX:
				primitiveInfo = searchForParameterInBlock(&block, "box", _FLOAT, _FLOAT, _FLOAT);
				break;
			case _INF_CYLINDER:
				break;
			case _INF_CONE:
				break;
			case _DISC:
				break;
			case _TRIANGLE:
				break;
			case _SQUARE:
				break;
			case _PLANE:
				primitiveInfo = searchForParameterInBlock(&block, "plane", _FLOAT, _FLOAT);
				break;
			case _OBJ:
				objPath = searchForParameterInBlock(&block, "obj", _STRING).stringParameters[0];
				break;
			default:
				break;
			}

			ParameterBlock ambientTextureInfo = searchForParameterInBlock(&block, "ambient_texture", _STRING, _FLOAT, _FLOAT);

			TransformInformation transInfo = getTransformInformation(&block);

			ParameterBlock baseColorInfo = searchForParameterInBlock(&block, "base_color", _INTEGER, _INTEGER, _INTEGER);
			ParameterBlock specularColorInfo = searchForParameterInBlock(&block, "specular_color", _INTEGER, _INTEGER, _INTEGER);
			float shininess = searchForParameterInBlock(&block, "shininess", _FLOAT).floatParameters[0];

			//ParameterBlock ambientInfo = searchForParameterInBlock(&block, "ambient_coeff", _FLOAT, _FLOAT, _FLOAT);
			//Vector3 ambient_coeff = Vector3(ambientInfo.floatParameters[0], ambientInfo.floatParameters[1], ambientInfo.floatParameters[2]);
			//
			//ParameterBlock difuseInfo = searchForParameterInBlock(&block, "difuse_coeff", _FLOAT, _FLOAT, _FLOAT);
			//Vector3 difuse_coeff = Vector3(difuseInfo.floatParameters[0], difuseInfo.floatParameters[1], difuseInfo.floatParameters[2]);
			//
			//ParameterBlock specInfo = searchForParameterInBlock(&block, "spec_coeff", _FLOAT, _FLOAT, _FLOAT);
			//Vector3 spec_coeff = Vector3(specInfo.floatParameters[0], specInfo.floatParameters[1], specInfo.floatParameters[2]);
			//
			//ParameterBlock ambientTextureInfo = searchForParameterInBlock(&block, "ambient_texture", _STRING, _FLOAT, _FLOAT);
			//std::string ambient_texture_filename = ambientTextureInfo.stringParameters[0];
			//Vector2 ambient_texture_UV_Scale = Vector2(ambientTextureInfo.floatParameters[0], ambientTextureInfo.floatParameters[1]);
			//
			//ParameterBlock difuseTextureInfo = searchForParameterInBlock(&block, "difuse_texture", _STRING, _FLOAT, _FLOAT);
			//std::string difuse_texture_filename = difuseTextureInfo.stringParameters[0];
			//Vector2 difuse_texture_UV_Scale = Vector2(difuseTextureInfo.floatParameters[0], difuseTextureInfo.floatParameters[1]);
			//
			//ParameterBlock specTextureInfo = searchForParameterInBlock(&block, "spec_texture", _STRING, _FLOAT, _FLOAT);
			//std::string spec_texture_filename = specTextureInfo.stringParameters[0];
			//Vector2 spec_texture_UV_Scale = Vector2(specTextureInfo.floatParameters[0], specTextureInfo.floatParameters[1]);

			float alpha = searchForParameterInBlock(&block, "alpha", _FLOAT).floatParameters[0];

			std::cout << "Read Object Block \n";

			Geometry object;

			object.SetPosition(transInfo._translate);
			object.SetRotation(transInfo._rotate);
			object.SetScale(transInfo._scale);

			object.SetMaterialBaseColor(Color((byte)baseColorInfo.intParameters[0],
				(byte)baseColorInfo.intParameters[1],
				(byte)baseColorInfo.intParameters[2]));
			object.SetMaterialSpecularColor(Color((byte)specularColorInfo.intParameters[0],
				(byte)specularColorInfo.intParameters[1],
				(byte)specularColorInfo.intParameters[2]));
			object.SetMaterialShininess(shininess);

			objects->push_back(object);

			switch (objType)
			{
			case _UNKNOWN_OBJECT:
				break;
			case _SPHERE:
				break;
			case _BOX:
				objects->back().CreateBoxMesh(primitiveInfo.floatParameters[0], primitiveInfo.floatParameters[1], primitiveInfo.floatParameters[2]);
				break;
			case _INF_CYLINDER:
				break;
			case _INF_CONE:
				break;
			case _DISC:
				break;
			case _TRIANGLE:
				break;
			case _SQUARE:
				break;
			case _PLANE:
				objects->back().CreatePlaneMesh(primitiveInfo.floatParameters[0], primitiveInfo.floatParameters[1]);
				break;
			case _OBJ:
				objPath = PathUtils::MakeAbsolutePath(objPath);
				OBJLoader::LoadOBJ((char*)objPath.c_str(), objects->back().GetMesh());
				break;
			default:
				break;
			}
		}
	}
}

ParameterBlock ConfigReader::searchForParameterInBlock(Block* block, std::string paramName, ParameterType param1, ParameterType param2, ParameterType param3)
{
	bool foundParameter = false;

	ParameterBlock params;

	int intParamCounter    = 0;
	int floatParamCounter  = 0;
	int stringParamCounter = 0;

	for(int i = 0; i < block->tokens.size(); i++)
	{
		if(block->tokens[i] == paramName)
		{
			switch (param1)
			{
				case _NONE:		break;
				case _FLOAT:   params.floatParameters[floatParamCounter]   = ::atof(block->tokens[i + 1].c_str());	floatParamCounter  += 1;	break;
				case _INTEGER: params.intParameters[intParamCounter]	   = ::atoi(block->tokens[i + 1].c_str());	intParamCounter	   += 1;	break;
				case _STRING:  params.stringParameters[stringParamCounter] = block->tokens[i + 1];					stringParamCounter += 1;	break;
			}

			switch (param2)
			{
				case _NONE:		break;
				case _FLOAT:   params.floatParameters[floatParamCounter]   = ::atof(block->tokens[i + 2].c_str());	floatParamCounter  += 1;	break;
				case _INTEGER: params.intParameters[intParamCounter]	   = ::atoi(block->tokens[i + 2].c_str());	intParamCounter    += 1;	break;
				case _STRING:  params.stringParameters[stringParamCounter] = block->tokens[i + 2];					stringParamCounter += 1;	break;
			}

			switch (param3)
			{
				case _NONE:		break;
				case _FLOAT:   params.floatParameters[floatParamCounter]   = ::atof(block->tokens[i + 3].c_str());	floatParamCounter  += 1;	break;
				case _INTEGER: params.intParameters[intParamCounter]	   = ::atoi(block->tokens[i + 3].c_str());	intParamCounter    += 1;	break;
				case _STRING:  params.stringParameters[stringParamCounter] = block->tokens[i + 3];					stringParamCounter += 1;	break;
			}

			foundParameter = true;

			break;
		}
	}

	if(!foundParameter)
	{
		assignDefaultParameterValue(&params, paramName);
	}

	return params;
}

TransformInformation ConfigReader::getTransformInformation(Block* block)
{
	TransformInformation transform;
	transform._rotate.x = searchForParameterInBlock(block, "rotateX", _FLOAT).floatParameters[0];
	transform._rotate.y = searchForParameterInBlock(block, "rotateY", _FLOAT).floatParameters[0];
	transform._rotate.z = searchForParameterInBlock(block, "rotateZ", _FLOAT).floatParameters[0];

	ParameterBlock transInfo = searchForParameterInBlock(block, "translate", _FLOAT, _FLOAT, _FLOAT);
	transform._translate.x = transInfo.floatParameters[0];
	transform._translate.y = transInfo.floatParameters[1];
	transform._translate.z = transInfo.floatParameters[2];

	ParameterBlock scaleInfo = searchForParameterInBlock(block, "scale", _FLOAT, _FLOAT, _FLOAT);
	transform._scale.x = scaleInfo.floatParameters[0];
	transform._scale.y = scaleInfo.floatParameters[1];
	transform._scale.z = scaleInfo.floatParameters[2];

	return transform;
}

ConfigObjectType ConfigReader::getObjectType(Block* block)
{
	ConfigObjectType type = _UNKNOWN_OBJECT;

	if(block->tokens[1] == "sphere")
		type = _SPHERE;
	else if(block->tokens[1] == "box")
		type = _BOX;
	else if(block->tokens[1] == "inf_cylinder")
		type = _INF_CYLINDER;
	else if(block->tokens[1] == "inf_cone")
		type = _INF_CONE;
	else if(block->tokens[1] == "disc")
		type = _DISC;
	else if(block->tokens[1] == "triangle")
		type = _TRIANGLE;
	else if(block->tokens[1] == "square")
		type = _SQUARE;
	else if(block->tokens[1] == "plane")
		type = _PLANE;
	else if (block->tokens[1] == "obj")
		type = _OBJ;

	if(type == _UNKNOWN_OBJECT)
	{
		std::cout << "Object Type Uknown, check spelling \n";
		system("pause");	
	}

	return type;
}

ConfigLightType ConfigReader::getLightType(Block* block)
{
	ConfigLightType type = _UNKNOWN_LIGHT;

	if(block->tokens[1] == "point")
		type = _POINT;
	else if(block->tokens[1] == "spot")
		type = _SPOT;
	else if(block->tokens[1] == "directional")
		type = _DIRECTIONAL;

	if(type == _UNKNOWN_LIGHT)
	{
		std::cout << "light Type Uknown, check spelling \n";
		system("pause");	
	}

	return type;
}

ConfigCameraType ConfigReader::getCameraType(Block* block)
{
	ConfigCameraType type;

	if(block->tokens[1] == "isometric")
		type = _ORTHOGRAPHIC;
	else if(block->tokens[1] == "perspective")
		type = _PERSPECTIVE;

	if(type == _UNKNOWN_CAMERA)
	{
		std::cout << "camera Type Uknown, check spelling \n"; 
		system("pause");	 
	}

	return type;
}

void ConfigReader::assignDefaultParameterValue(ParameterBlock* params, std::string paramName)
{
	// model
	if(paramName == "ambient")
	{
		params->intParameters[0] = 51;
		params->intParameters[1] = 51;
		params->intParameters[2] = 51;
	}
	else if(paramName == "depth")
	{
		params->intParameters[0] = 0;
	}
	else if(paramName == "threshold")
	{
		params->floatParameters[0] = 0.f;
	}
	
	//camera
	else if(paramName == "perspective")
	{
		params->floatParameters[0] = 45.f;
	}
	else if (paramName == "isometric")
	{
		params->floatParameters[0] = 10.f;
	}
	else if(paramName == "width")
	{
		params->intParameters[0] = 1280;
	}
	else if(paramName == "height")
	{
		params->intParameters[0] = 720;
	}
	else if (paramName == "background_color")
	{
		params->intParameters[0] = 0;
		params->intParameters[1] = 0;
		params->intParameters[2] = 0;
	}

	//lights
	else if(paramName == "spot")
	{
		params->floatParameters[0] = 0.f;
	}
	else if(paramName == "intensity")
	{
		params->floatParameters[0] = 1.f;
	}
	else if(paramName == "color")
	{
		params->intParameters[0] = 255;
		params->intParameters[1] = 255;
		params->intParameters[2] = 255;
	}
	
	// objects
	if (paramName == "base_color")
	{
		params->intParameters[0] = 255;
		params->intParameters[1] = 255;
		params->intParameters[2] = 255;
	}
	else if (paramName == "specular_color")
	{
		params->intParameters[0] = 255;
		params->intParameters[1] = 255;
		params->intParameters[2] = 255;
	}
	else if (paramName == "shininess")
	{
		params->floatParameters[0] = 0.5f;
	}
	else if(paramName == "ambient_coeff")
	{
		params->floatParameters[0] = 0.f;
		params->floatParameters[1] = 0.f;
		params->floatParameters[2] = 0.f;
	}
	else if(paramName == "difuse_coeff")
	{
		params->floatParameters[0] = 0.f;
		params->floatParameters[1] = 0.f;
		params->floatParameters[2] = 0.f;
	}
	else if(paramName == "spec_coeff")
	{
		params->floatParameters[0] = 0.f;
		params->floatParameters[1] = 0.f;
		params->floatParameters[2] = 0.f;
	}
	else if(paramName == "ambient_texture")
	{
		params->floatParameters[0] = 0.f;
		params->floatParameters[1] = 0.f;
		params->stringParameters[0] = "default_ambient_tex.jpg";
	}
	else if(paramName == "difuse_texture")
	{
		params->floatParameters[0] = 0.f;
		params->floatParameters[1] = 0.f;
		params->stringParameters[0] = "default_difuse_tex.jpg";
	}
	else if(paramName == "spec_texture")
	{
		params->floatParameters[0] = 0.f;
		params->floatParameters[1] = 0.f;
		params->stringParameters[0] = "default_spec_tex.jpg";
	}
	else if(paramName == "alpha")
	{
		params->floatParameters[0] = 0.f;
	}

	// Transform values for Cameras, Lights & Objects
	else if(paramName == "rotateX")
	{
		params->floatParameters[0] = 0.f;
	}	
	else if(paramName == "rotateY")
	{
		params->floatParameters[0] = 0.f;
	}	
	else if(paramName == "rotateZ")
	{
		params->floatParameters[0] = 0.f;
	}
	else if(paramName == "translate")
	{
		params->floatParameters[0] = 0.f;
		params->floatParameters[1] = 0.f;
		params->floatParameters[2] = 0.f;
	}
	else if(paramName == "scale")
	{
		params->floatParameters[0] = 1.f;
		params->floatParameters[1] = 1.f;
		params->floatParameters[2] = 1.f;
	}
}