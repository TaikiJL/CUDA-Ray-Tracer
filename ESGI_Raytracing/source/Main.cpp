#define NOMINMAX

#include <iostream>
#include <Windows.h>
#include "GL/freeglut.h"
#include <string>

#include "Scene.h"
#include "Bitmap.h"
#include "Camera.h"
#include "Light.h"
#include "MyMath.h"
#include "Vector.h"
#include "OBJLoader.h"
#include "Mesh.h"
#include "ConfigReader.h"

extern "C" void Render(Color *pixelArray, const Camera &camera);

// OpenGL functions
void Initialize(int argc, char* argv[]);
void InitWindow(int iWindowWidth, int iWindowHeight);
void ResizeFunction(int newWidth, int newHeight);
void RenderFunction(void);
void KeyboardSpecialFunction(int key, int x, int y);
void Cleanup(int errorCode, bool bExit = true);
void LoadRenderTexture();

static int defaultDisplayWidth;
static int defaultDisplayHeight;

static int glWindowWidth, glWindowHeight;
static int windowHandle = 0;

static GLuint renderTexture;

static Color* pixelArray;

int main(int argc, char *argv[])
{
	std::cout << "Please enter the path to your configuration file:" << std::endl;
	std::string userPath;

	std::getline(std::cin, userPath);

	if (userPath.empty())
		userPath = "scene.conf";

	ConfigReader configReader((char*)userPath.c_str());
	configReader.readConfig(&Scene::GetLights(), &Scene::GetCameras(), &Scene::GetGeometries());

	// This is used to avoid errors when looking for &Scene::GetLights()[0] in Renderer.cu (to do the same for Geometries)
	if (Scene::GetLights().size() == 0)
	{
		Light emptyLight;
		emptyLight.SetIntensity(0.f);
		Scene::GetLights().push_back(emptyLight);
	}

	Scene::SetMainCamera(Scene::GetCameras()[0]); // Hard-coding: main camera is the first camera of the vector

	int width = Scene::GetRenderWidth();
	int height = Scene::GetRenderHeight();
	
	defaultDisplayWidth = width;
	defaultDisplayHeight = height;

	int totalPixelCount = width * height;
	
	pixelArray = new Color[totalPixelCount];

	Initialize(argc, argv);
	InitWindow(defaultDisplayWidth, defaultDisplayHeight);

	glutMainLoop();
	
	Cleanup(0);

	exit(EXIT_SUCCESS);
}

void Initialize(int argc, char* argv[])
{
	glutInit(&argc, argv);

	//std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
	glWindowWidth = defaultDisplayWidth;
	glWindowHeight = defaultDisplayHeight;
}

void InitWindow(int iWindowWidth, int iWindowHeight)
{
	//glutInitContextVersion(4, 0);
	//glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
	//glutInitContextProfile(GLUT_CORE_PROFILE);

	glutSetOption(
		GLUT_ACTION_ON_WINDOW_CLOSE,
		GLUT_ACTION_GLUTMAINLOOP_RETURNS
		);

	glutInitWindowSize(iWindowWidth, iWindowHeight);

	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);

	windowHandle = glutCreateWindow("RayTracer");

	if (windowHandle < 1) {
		fprintf(
			stderr,
			"ERROR: Could not create a new rendering window.\n"
			);
		exit(EXIT_FAILURE);
	}

	glGenTextures(1, &renderTexture);

	Render(pixelArray, Scene::GetMainCamera());
	LoadRenderTexture();

	glutReshapeFunc(ResizeFunction);
	glutDisplayFunc(RenderFunction);
	glutSpecialFunc(KeyboardSpecialFunction);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

void ResizeFunction(int newWidth, int newHeight)
{
	glViewport(0, 0, newWidth, newHeight);

	glWindowWidth = newWidth;
	glWindowHeight = newHeight;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(-1.f, 1.f, -1.f, 1.f);

	glutPostRedisplay();
}

void RenderFunction(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	Render(pixelArray, Scene::GetMainCamera());
	LoadRenderTexture();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, renderTexture);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0); glVertex2f(-1.f, -1.f);
	glTexCoord2f(1.0, 0.0); glVertex2f(1.f, -1.f);
	glTexCoord2f(1.0, 1.0); glVertex2f(1.f, 1.f);
	glTexCoord2f(0.0, 1.0); glVertex2f(-1.f, 1.f);
	glEnd();

	glDisable(GL_TEXTURE_2D);

	glutSwapBuffers();
	glutPostRedisplay();
}

void KeyboardSpecialFunction(int key, int x, int y)
{
	int mod = glutGetModifiers();
	
	if (mod != 4)
	{
		switch (key)
		{
		case GLUT_KEY_LEFT:
			Scene::GetMainCamera().Translate(Vector3(-1.f, 0.f, 0.f));
			break;
		case GLUT_KEY_RIGHT:
			Scene::GetMainCamera().Translate(Vector3(1.f, 0.f, 0.f));
			break;
		case GLUT_KEY_UP:
			Scene::GetMainCamera().Translate(Vector3(0.f, 0.f, 1.f));
			break;
		case GLUT_KEY_DOWN:
			Scene::GetMainCamera().Translate(Vector3(0.f, 0.f, -1.f));
			break;
		default:
			break;
		}
	}
	else
	{
		switch (key)
		{
		case GLUT_KEY_LEFT:
			Scene::GetMainCamera().SetRotation(Scene::GetMainCamera().GetRotation() + Vector3(0.f, -1.f, 0.f));
			break;
		case GLUT_KEY_RIGHT:
			Scene::GetMainCamera().SetRotation(Scene::GetMainCamera().GetRotation() + Vector3(0.f, 1.f, 0.f));
			break;
		case GLUT_KEY_UP:
			Scene::GetMainCamera().SetRotation(Scene::GetMainCamera().GetRotation() + Vector3(1.f, 0.f, 0.f));
			break;
		case GLUT_KEY_DOWN:
			Scene::GetMainCamera().SetRotation(Scene::GetMainCamera().GetRotation() + Vector3(-1.f, 0.f, 0.f));
			break;
		default:
			break;
		}
	}
	
}

void Cleanup(int errorCode, bool bExit)
{
	delete[] pixelArray;

	for (int i = 0; i < Scene::GetGeometries().size(); i++)
	{
		Scene::GetGeometries()[i].GetMesh().ClearMesh();
	}

	if (windowHandle != 0)
	{
		glutDestroyWindow(windowHandle);
		windowHandle = 0;
	}

	if (bExit)
	{
		exit(errorCode);
	}
}

void LoadRenderTexture()
{
	glBindTexture(GL_TEXTURE_2D, renderTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, Scene::GetRenderWidth(), Scene::GetRenderHeight(), 0, GL_RGBA, GL_FLOAT, pixelArray);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}
