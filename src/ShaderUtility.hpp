#pragma once

#include <GL/glew.h>
#include <string>

namespace ShaderUtility {

// Compiles and links a vertex + fragment shader pair. Throws on failure.
GLuint createProgram(const std::string &vert_path,
                     const std::string &frag_path);

} // namespace ShaderUtility
