#include "Global.hpp"
#include "GranularParticles.hpp"
#include "GranularSystem.hpp"
#include "ShaderUtility.hpp"
#include "VBO.hpp"
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>

// vbo and GL variables
static GLuint particlesVBO;
static GLuint particlesColorVBO;
static GLuint m_particles_program;
static const int m_window_h = 1600;
static const int m_fov = 30;
static const float particle_radius = 0.01f;

// view variables
static float rot[2] = {0.0f, 0.0f};
static int mousePos[2] = {-1, -1};
static bool mouse_left_down = false;
static float zoom = 0.3f;

// state variables
static int frameId = 0;
static float totalTime = 0.0f;
bool running = false;

// particle system variables
std::shared_ptr<GranularSystem> p_system;
const float3 space_size = make_float3(1.0f);
const float dt = 0.002f;
const float3 G = make_float3(0.0f, -9.8f, 0.0f);
const float sphSpacing = 0.02f;
const float smoothing_radius = 2.0f * sphSpacing;
const float cell_length = 1.01f * smoothing_radius;
const int3 cell_size = make_int3(ceil(space_size.x / cell_length),
                                 ceil(space_size.y / cell_length),
                                 ceil(space_size.z / cell_length));

void init_granular_system() {
  // NOTE: Fill up the initial positions of the particles
  std::vector<float3> pos;

  for (auto i = 0; i < 36; ++i) {
    for (auto j = 0; j < 24; ++j) {
      for (auto k = 0; k < 24; ++k) {
        auto x = make_float3(0.27f + sphSpacing * j, 0.10f + sphSpacing * i,
                             0.27f + sphSpacing * k);
        pos.push_back(x);
      }
    }
  }
  auto granular_particles = std::make_shared<GranularParticles>(pos);
  pos.clear();

  const auto compact_size = 2 * make_int3(ceil(space_size.x / cell_length),
                                          ceil(space_size.y / cell_length),
                                          ceil(space_size.z / cell_length));
  // front and back
  for (auto i = 0; i < compact_size.x; ++i) {
    for (auto j = 0; j < compact_size.y; ++j) {
      auto x = make_float3(i, j, 0) / make_float3(compact_size - make_int3(1)) *
               space_size;
      pos.push_back(0.99f * x + 0.005f * space_size);
      x = make_float3(i, j, compact_size.z - 1) /
          make_float3(compact_size - make_int3(1)) * space_size;
      pos.push_back(0.99f * x + 0.005f * space_size);
    }
  }
  // top and bottom
  for (auto i = 0; i < compact_size.x; ++i) {
    for (auto j = 0; j < compact_size.z - 2; ++j) {
      auto x = make_float3(i, 0, j + 1) /
               make_float3(compact_size - make_int3(1)) * space_size;
      pos.push_back(0.99f * x + 0.005f * space_size);
      x = make_float3(i, compact_size.y - 1, j + 1) /
          make_float3(compact_size - make_int3(1)) * space_size;
      pos.push_back(0.99f * x + 0.005f * space_size);
    }
  }
  // left and right
  for (auto i = 0; i < compact_size.y - 2; ++i) {
    for (auto j = 0; j < compact_size.z - 2; ++j) {
      auto x = make_float3(0, i + 1, j + 1) /
               make_float3(compact_size - make_int3(1)) * space_size;
      pos.push_back(0.99f * x + 0.005f * space_size);
      x = make_float3(compact_size.x - 1, i + 1, j + 1) /
          make_float3(compact_size - make_int3(1)) * space_size;
      pos.push_back(0.99f * x + 0.005f * space_size);
    }
  }

  auto boundary_particles = std::make_shared<GranularParticles>(pos);
  p_system = std::make_shared<GranularSystem>(granular_particles,
                                              boundary_particles, space_size,
                                              cell_length, dt, G, cell_size);
}

void createVBO(GLuint *vbo, const unsigned int length) {
  // create buffer object
  glGenBuffers(1, vbo);
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);

  // initialize buffer object
  glBufferData(GL_ARRAY_BUFFER, length, nullptr, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // register buffer object with CUDA
  CUDA_CALL(cudaGLRegisterBufferObject(*vbo));
}

void deleteVBO(GLuint *vbo) {
  glBindBuffer(1, *vbo);
  glDeleteBuffers(1, vbo);

  CUDA_CALL(cudaGLUnregisterBufferObject(*vbo));

  *vbo = NULL;
}

void onClose(void) {
  deleteVBO(&particlesVBO);
  deleteVBO(&particlesColorVBO);

  p_system = nullptr;
  CUDA_CALL(cudaDeviceReset());
  exit(0);
}

namespace particle_attributes {
enum {
  POSITION,
  COLOR,
  SIZE,
};
}

void mouseFunc(const int button, const int state, const int x, const int y) {
  if (GLUT_DOWN == state) {
    if (GLUT_LEFT_BUTTON == button) {
      mouse_left_down = true;
      mousePos[0] = x;
      mousePos[1] = y;
    } else if (GLUT_RIGHT_BUTTON == button) {
    }
  } else {
    mouse_left_down = false;
  }
  return;
}

void motionFunc(const int x, const int y) {
  int dx, dy;
  if (-1 == mousePos[0] && -1 == mousePos[1]) {
    mousePos[0] = x;
    mousePos[1] = y;
    dx = dy = 0;
  } else {
    dx = x - mousePos[0];
    dy = y - mousePos[1];
  }
  if (mouse_left_down) {
    rot[0] += (float(dy) * 180.0f) / 720.0f;
    rot[1] += (float(dx) * 180.0f) / 720.0f;
  }

  mousePos[0] = x;
  mousePos[1] = y;

  glutPostRedisplay();
  return;
}

extern "C" void
generate_dots(float3 *dot, float3 *color,
              const std::shared_ptr<GranularParticles> particles);

void renderParticles(void) {
  // map OpenGL buffer object for writing from CUDA
  float3 *dptr;
  float3 *cptr;
  CUDA_CALL(cudaGLMapBufferObject((void **)&dptr, particlesVBO));
  CUDA_CALL(cudaGLMapBufferObject((void **)&cptr, particlesColorVBO));

  // calculate the dots' position and color
  generate_dots(dptr, cptr, p_system->get_particles());

  // unmap buffer object
  CUDA_CALL(cudaGLUnmapBufferObject(particlesVBO));
  CUDA_CALL(cudaGLUnmapBufferObject(particlesColorVBO));

  glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
  glVertexPointer(3, GL_FLOAT, 0, nullptr);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, particlesColorVBO);
  glColorPointer(3, GL_FLOAT, 0, nullptr);
  glEnableClientState(GL_COLOR_ARRAY);

  glDrawArrays(GL_POINTS, 0, p_system->size());

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
  return;
}

// TODO: add step
void one_step() {
  ++frameId;
  p_system->step();
  // TODO fix
  // const auto milliseconds = p_system->step();
  // totalTime += milliseconds;
  // printf("Frame %d - %2.2f ms, avg time - %2.2f ms/frame (%3.2f FPS)\r",
  // 	frameId%10000, milliseconds, totalTime / float(frameId),
  // float(frameId)*1000.0f/totalTime);
}

void initGL(void) {
  // create VBOs
  createVBO(&particlesVBO, sizeof(float3) * p_system->size());
  createVBO(&particlesColorVBO, sizeof(float3) * p_system->size());
  // initiate shader program
  m_particles_program = glCreateProgram();
  glBindAttribLocation(m_particles_program, particle_attributes::SIZE,
                       "pointSize");
  ShaderUtility::attachAndLinkProgram(
      m_particles_program,
      ShaderUtility::loadShaders("/home/bliss/Documents/gpu_projects/"
                                 "granular_gpu/src/shaders/particles.vert",
                                 "/home/bliss/Documents/gpu_projects/"
                                 "granular_gpu/src/shaders/particles.frag"));
  return;
}

static void displayFunc(void) {
  if (running) {
    one_step();
  }

  glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // ----------------------------------------------------------------
  // We DO NOT reset the projection matrix here anymore!
  // The reshape() callback now handles the correct aspect ratio.
  // ----------------------------------------------------------------

  // Set up the camera in ModelView

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(0, 0, 1.0 / zoom, 0, 0, 0, 0, 1, 0);

  // Some example transformations
  glPushMatrix();
  glRotatef(rot[0], 1.0f, 0.0f, 0.0f);
  glRotatef(rot[1], 0.0f, 1.0f, 0.0f);

  // Draw a wire cube for reference
  glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
  glLineWidth(2.0);
  glEnable(GL_MULTISAMPLE_ARB);
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  glutSolidCube(1.0);

  // ----------------------------------------------------------------
  // Render Particles
  // ----------------------------------------------------------------
  glUseProgram(m_particles_program);
  // Scale for point sprites
  float uniformVal = m_window_h / tanf(m_fov * 0.5f * float(M_PI) / 180.0f);
  glUniform1f(glGetUniformLocation(m_particles_program, "pointScale"),
              uniformVal);
  glUniform1f(glGetUniformLocation(m_particles_program, "pointRadius"),
              particle_radius);

  glEnable(GL_POINT_SPRITE_ARB);
  glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);

  glDepthMask(GL_TRUE);
  glEnable(GL_DEPTH_TEST);

  glPushMatrix();
  glTranslatef(-0.5f, -0.5f, -0.5f);
  renderParticles();
  glPopMatrix();
  glPopMatrix();

  glutSwapBuffers();
  glutPostRedisplay();
}

void keyboardFunc(const unsigned char key, const int x, const int y) {
  switch (key) {
  case '1':
    frameId = 0;
    totalTime = 0.0f;
    break;
  case ' ':
    running = !running;
    break;
  case ',':
    zoom *= 1.2f;
    break;
  case '.':
    zoom /= 1.2f;
    break;
  case 'q':
  case 'Q':
    onClose();
    break;
  case 'r':
  case 'R':
    rot[0] = rot[1] = 0;
    zoom = 0.3f;
    break;
  case 'n':
  default:;
  }
}

void reshape(int width, int height) {
  if (height == 0)
    height = 1; // Prevent divide-by-zero errors

  // Update the viewport to match the window dimensions
  glViewport(0, 0, width, height);

  // Set up the projection matrix to preserve aspect ratio
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  float aspect = (float)width / (float)height; // Calculate aspect ratio
  gluPerspective(45.0, aspect, 0.1, 100.0);    // FOV, aspect ratio, near, far

  glMatrixMode(GL_MODELVIEW); // Return to modelview matrix
}

int main(int argc, char *argv[]) {
  try {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_MULTISAMPLE);

    glutInitWindowPosition(400, 0);
    // Get screen dimensions
    int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
    int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);

    // Calculate centered position
    int posX = (screenWidth - m_window_h) / 2;
    int posY = (screenHeight - m_window_h) / 2;

    // Create a window at the center of the screen
    glutInitWindowPosition(posX, posY);
    glutInitWindowSize(m_window_h, m_window_h);
    // glutInitWindowSize(m_window_h, m_window_h);
    glutCreateWindow("");

    glutFullScreen();
    glutDisplayFunc(&displayFunc);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(&keyboardFunc);
    glutMouseFunc(&mouseFunc);
    glutMotionFunc(&motionFunc);

    glewInit();
    ////////////////////
    init_granular_system();
    initGL();

    std::cout << "Instructions\n";
    std::cout << "Controls\n";
    std::cout << "Space - Start/Pause\n";
    std::cout << "Key N - One Step Forward\n";
    std::cout << "Key Q - Quit\n";
    std::cout << "Key R - Reset Viewpoint\n";
    std::cout << "Key , - Zoom In\n";
    std::cout << "Key . - Zoom Out\n";
    std::cout << "Mouse Drag - Change Viewpoint\n\n";
    ////////////////////
    glutMainLoop();
  } catch (...) {
    std::cout << "Unknown Exception at " << __FILE__ << ": line " << __LINE__
              << "\n";
  }
  return 0;
}
