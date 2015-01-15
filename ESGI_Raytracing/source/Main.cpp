#include <iostream>
#include <Windows.h>
#include "GL/freeglut.h"

#include "Scene.h"
#include "Bitmap.h"
#include "Camera.h"
#include "Sphere.h"
#include "Light.h"
#include "MyMath.h"
#include "Vector.h"

extern "C" void Render(Color *pixelArray, const Camera &camera);

// OpenGL functions
void Initialize(int argc, char* argv[]);
void InitWindow(int iWindowWidth, int iWindowHeight);
void ResizeFunction(int newWidth, int newHeight);
void RenderFunction(void);
void KeyboardSpecialFunction(int key, int x, int y);
void Cleanup(int errorCode, bool bExit = true);
void LoadRenderTexture();

const static int defaultDisplayWidth = 1280;
const static int defaultDisplayHeight = 720;

static int glWindowWidth, glWindowHeight;
static int windowHandle = 0;

static GLuint renderTexture;

static Color* pixelArray;

int main(int argc, char *argv[])
{
	int width = Scene::GetRenderWidth();
	int height = Scene::GetRenderHeight();

	Bitmap img(width, height);
	
	int totalPixelCount = width*height;

	pixelArray = new Color[totalPixelCount];

	Camera cam = Camera();
	//cam.SetBackgroundColor(Color((byte)49, 77, 121));
	cam.SetBackgroundColor(Color(0.f));
	Scene::SetMainCamera(cam);

	Sphere sphere = Sphere();
	sphere.SetPosition(Vector3(0.f, 0.f, 20.f));
	sphere.SetMaterialBaseColor(Color(0.8f));
	sphere.SetMaterialShininess(0.8f);
	Scene::AddGeometry(sphere);

	Sphere sphere3 = Sphere();
	sphere3.SetPosition(Vector3(-3.f, 0.f, 20.f));
	sphere3.SetMaterialBaseColor(Color(1.f, 0.f, 0.f));
	sphere3.SetMaterialShininess(0.3f);
	Scene::AddGeometry(sphere3);

	Sphere sphere2 = Sphere();
	sphere2.SetPosition(Vector3(3.f, 0.f, 20.f));
	sphere2.SetMaterialBaseColor(Color(0.f, 0.f, 1.f));
	sphere2.SetMaterialShininess(0.3f);
	Scene::AddGeometry(sphere2);

	Light light = Light();
	light.SetPosition(Vector3(0.f, 15.f, 10.f));
	Scene::AddLight(light);

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
	switch (key)
	{
	case GLUT_KEY_LEFT:
		Scene::GetGeometries()[0].Translate(Vector3(-0.1f, 0.f, 0.f));
		break;
	case GLUT_KEY_RIGHT:
		Scene::GetGeometries()[0].Translate(Vector3(0.1f, 0.f, 0.f));
		break;
	case GLUT_KEY_UP:
		Scene::GetLights()[0].Translate(Vector3(0.f, 1.f, 0.f));
		break;
	case GLUT_KEY_DOWN:
		Scene::GetLights()[0].Translate(Vector3(0.f, -1.f, 0.f));
		break;
	default:
		break;
	}
}

void Cleanup(int errorCode, bool bExit)
{
	delete[] pixelArray;

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
