#pragma once

#include <GL/glew.h>

namespace ShaderUtility {

typedef struct {
  unsigned int vertex;
  unsigned int fragment;
} shaders_t;

void *initGLEW();

shaders_t loadShaders(char *vert_path, char *frag_path);

void attachAndLinkProgram(unsigned int program, shaders_t shaders);

char *loadFile(char *fname, int &fSize);

// printShaderInfoLog
// From OpenGL Shading Language 3rd Edition, p215-216
// Display (hopefully) useful error messages if shader fails to compile
void printShaderInfoLog(int shader);

void printLinkInfoLog(int prog);

} // namespace ShaderUtility
