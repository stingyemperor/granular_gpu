#include "ShaderUtility.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ShaderUtility {
namespace {

std::string loadFile(const std::string &path) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("unable to open shader file: " + path);
  }
  std::stringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

GLuint compileShader(GLenum type, const std::string &path) {
  const std::string source = loadFile(path);
  const char *src = source.c_str();
  const GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &src, nullptr);
  glCompileShader(shader);

  GLint compiled = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    GLint len = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
    std::vector<GLchar> log(len + 1, '\0');
    glGetShaderInfoLog(shader, len, nullptr, log.data());
    glDeleteShader(shader);
    throw std::runtime_error("failed to compile " + path + ":\n" + log.data());
  }
  return shader;
}

} // namespace

GLuint createProgram(const std::string &vert_path,
                     const std::string &frag_path) {
  const GLuint vert = compileShader(GL_VERTEX_SHADER, vert_path);
  const GLuint frag = compileShader(GL_FRAGMENT_SHADER, frag_path);

  const GLuint program = glCreateProgram();
  glAttachShader(program, vert);
  glAttachShader(program, frag);
  glLinkProgram(program);
  glDeleteShader(vert);
  glDeleteShader(frag);

  GLint linked = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &linked);
  if (!linked) {
    GLint len = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
    std::vector<GLchar> log(len + 1, '\0');
    glGetProgramInfoLog(program, len, nullptr, log.data());
    glDeleteProgram(program);
    throw std::runtime_error(std::string("failed to link shaders:\n") +
                             log.data());
  }
  return program;
}

} // namespace ShaderUtility
