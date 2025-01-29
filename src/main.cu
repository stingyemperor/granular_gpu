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
#include <random>
#include <vector>

// vbo and GL variables
static GLuint particlesVBO;
static GLuint particlesColorVBO;
static GLuint upsampledParticlesVBO;
static GLuint upsampledParticlesColorVBO;
static const float upsampled_particle_radius = 0.005f; // Half
static GLuint m_particles_program;
static const int m_window_h = 1600;
static const int m_fov = 30;
static const float particle_radius = 0.01f;
static const float density = 238732.4146f;
static size_t last_particle_count =
    0; // Add this to track particle count changes

// view variables
static float rot[2] = {0.0f, 0.0f};
static int mousePos[2] = {-1, -1};
static bool mouse_left_down = false;
static float zoom = 0.3f;

// state variables
static int frameId = 0;
static float totalTime = 0.0f;
bool running = false;
bool show_surface = false;

// particle system variables
std::shared_ptr<GranularSystem> p_system;
const float3 space_size = make_float3(1.5f, 1.5f, 1.5f);
const float dt = 0.002f;
const float3 G = make_float3(0.0f, -9.8f, 0.0f);
const float sphSpacing = 0.02f;
const float initSpacing = 0.030f;
const float smoothing_radius = 2.0f * sphSpacing;
const float cell_length = 1.01f * smoothing_radius;
// const float cell_length = 0.02f;
const int3 cell_size = make_int3(ceil(space_size.x / cell_length),
                                 ceil(space_size.y / cell_length),
                                 ceil(space_size.z / cell_length));

void init_granular_system() {
  // NOTE: Fill up the initial positions of the particles
  std::vector<float3> pos;
  // 36 24 24
  for (auto i = 0; i < 45; ++i) {
    for (auto j = 0; j < 30; ++j) {
      for (auto k = 0; k < 30; ++k) {
        auto x = make_float3(0.27f + initSpacing * j, 0.13f + initSpacing * i,
                             0.17f + initSpacing * k);
        pos.push_back(x);
      }
    }
  }
  auto granular_particles = std::make_shared<GranularParticles>(pos);

  std::vector<float3> upsampled_pos;
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-particle_radius,
                                                     particle_radius);

  for (const auto &particle : pos) {
    for (int n = 0; n < 10;
         ++n) { // Generate 10 upsampled particles per original particle
      float3 offset =
          make_float3(distribution(generator), distribution(generator),
                      distribution(generator));
      upsampled_pos.push_back(particle + offset);
    }
  }

  auto upsampled_particles = std::make_shared<GranularParticles>(upsampled_pos);

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

  p_system = std::make_shared<GranularSystem>(
      granular_particles, boundary_particles, upsampled_particles, space_size,
      cell_length, dt, G, cell_size, density, upsampled_particle_radius);
}

void resizeVBO(GLuint *vbo, size_t new_size) {
  // Unregister old VBO from CUDA
  CUDA_CALL(cudaGLUnregisterBufferObject(*vbo));

  // Delete old VBO
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);
  glBufferData(GL_ARRAY_BUFFER, new_size * sizeof(float3), nullptr,
               GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Re-register with CUDA
  CUDA_CALL(cudaGLRegisterBufferObject(*vbo));
}

void createVBO(GLuint *vbo, const unsigned int size) {
  // Check if VBO already exists
  if (*vbo != 0) {
    // Unregister from CUDA if needed
    cudaError_t err = cudaGLUnregisterBufferObject(*vbo);
    if (err != cudaSuccess && err != cudaErrorInvalidResourceHandle) {
      std::cerr << "CUDA unregister error: " << cudaGetErrorString(err)
                << std::endl;
    }

    // Delete existing VBO
    glDeleteBuffers(1, vbo);
  }

  // create buffer object
  glGenBuffers(1, vbo);
  if (*vbo == 0) {
    throw std::runtime_error("Failed to create VBO");
  }

  glBindBuffer(GL_ARRAY_BUFFER, *vbo);
  glBufferData(GL_ARRAY_BUFFER, size * sizeof(float3), nullptr,
               GL_DYNAMIC_DRAW);

  // Check for OpenGL errors
  GLenum gl_error = glGetError();
  if (gl_error != GL_NO_ERROR) {
    std::cerr << "OpenGL error in createVBO: " << gl_error << std::endl;
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    throw std::runtime_error("OpenGL error in createVBO");
  }

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // register buffer object with CUDA
  cudaError_t cuda_err = cudaGLRegisterBufferObject(*vbo);
  if (cuda_err != cudaSuccess) {
    std::cerr << "CUDA register error: " << cudaGetErrorString(cuda_err)
              << std::endl;
    throw std::runtime_error("Failed to register VBO with CUDA");
  }
}

void deleteVBO(GLuint *vbo) {
  if (*vbo == 0)
    return;

  // Unregister from CUDA
  cudaError_t err = cudaGLUnregisterBufferObject(*vbo);
  if (err != cudaSuccess && err != cudaErrorInvalidResourceHandle) {
    std::cerr << "CUDA unregister error: " << cudaGetErrorString(err)
              << std::endl;
  }

  // Delete buffer
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);
  glDeleteBuffers(1, vbo);

  *vbo = 0;
}

void onClose(void) {
  deleteVBO(&particlesVBO);
  deleteVBO(&particlesColorVBO);
  deleteVBO(&upsampledParticlesVBO);
  deleteVBO(&upsampledParticlesColorVBO);

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
              const std::shared_ptr<GranularParticles> particles, int *surface,
              const float max_mass, const float min_mass, bool show_surface);

void renderParticles(void) {
  size_t current_particle_count = p_system->size();

  // Check if particle count has changed
  if (current_particle_count != last_particle_count) {
    std::cout << "Particle count changed from " << last_particle_count << " to "
              << current_particle_count << std::endl;

    try {
      // Resize VBOs if needed
      resizeVBO(&particlesVBO, current_particle_count);
      resizeVBO(&particlesColorVBO, current_particle_count);
      last_particle_count = current_particle_count;
    } catch (const std::exception &e) {
      std::cerr << "Error resizing VBOs: " << e.what() << std::endl;
      return;
    }
  }

  // map OpenGL buffer object for writing from CUDA
  float3 *dptr;
  float3 *cptr;

  cudaError_t err;

  err = cudaGLMapBufferObject((void **)&dptr, particlesVBO);
  if (err != cudaSuccess) {
    std::cerr << "Error mapping position VBO: " << cudaGetErrorString(err)
              << std::endl;
    return;
  }

  err = cudaGLMapBufferObject((void **)&cptr, particlesColorVBO);
  if (err != cudaSuccess) {
    cudaGLUnmapBufferObject(particlesVBO); // Clean up first buffer
    std::cerr << "Error mapping color VBO: " << cudaGetErrorString(err)
              << std::endl;
    return;
  }

  try {
    // calculate the dots' position and color
    generate_dots(dptr, cptr, p_system->get_particles(),
                  p_system->get_particles()->get_surface_ptr(),
                  p_system->get_max_mass(), p_system->get_min_mass(),
                  show_surface);
  } catch (const std::exception &e) {
    std::cerr << "Error in generate_dots: " << e.what() << std::endl;
  }

  // unmap buffer objects
  CUDA_CALL(cudaGLUnmapBufferObject(particlesVBO));
  CUDA_CALL(cudaGLUnmapBufferObject(particlesColorVBO));

  glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
  glVertexPointer(3, GL_FLOAT, 0, nullptr);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, particlesColorVBO);
  glColorPointer(3, GL_FLOAT, 0, nullptr);
  glEnableClientState(GL_COLOR_ARRAY);

  glDrawArrays(GL_POINTS, 0, current_particle_count);

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
}

void renderUpsampledParticles(void) {
  size_t current_upsampled_count = p_system->upsampled_size();

  // map OpenGL buffer object for writing from CUDA
  float3 *dptr;
  float3 *cptr;

  CUDA_CALL(cudaGLMapBufferObject((void **)&dptr, upsampledParticlesVBO));
  CUDA_CALL(cudaGLMapBufferObject((void **)&cptr, upsampledParticlesColorVBO));

  try {
    // calculate the dots' position and color
    generate_dots(dptr, cptr, p_system->get_upsampled(),
                  p_system->get_upsampled()->get_surface_ptr(),
                  p_system->get_max_mass(), p_system->get_min_mass(),
                  show_surface);
  } catch (const std::exception &e) {
    std::cerr << "Error in generate_dots for upsampled particles: " << e.what()
              << std::endl;
  }

  // unmap buffer objects
  CUDA_CALL(cudaGLUnmapBufferObject(upsampledParticlesVBO));
  CUDA_CALL(cudaGLUnmapBufferObject(upsampledParticlesColorVBO));

  glBindBuffer(GL_ARRAY_BUFFER, upsampledParticlesVBO);
  glVertexPointer(3, GL_FLOAT, 0, nullptr);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, upsampledParticlesColorVBO);
  glColorPointer(3, GL_FLOAT, 0, nullptr);
  glEnableClientState(GL_COLOR_ARRAY);

  glDrawArrays(GL_POINTS, 0, current_upsampled_count);

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
}

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

  // New VBOs for upsampled particles
  createVBO(&upsampledParticlesVBO,
            sizeof(float3) * p_system->upsampled_size());
  createVBO(&upsampledParticlesColorVBO,
            sizeof(float3) * p_system->upsampled_size());

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

  // First draw text
  glDisable(GL_DEPTH_TEST);
  glMatrixMode(GL_PROJECTION);
  glPushMatrix(); // Save current projection matrix
  glLoadIdentity();
  glOrtho(0, glutGet(GLUT_WINDOW_WIDTH), 0, glutGet(GLUT_WINDOW_HEIGHT), -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix(); // Save current modelview matrix
  glLoadIdentity();

  glColor3f(0.0f, 0.0f, 0.0f);
  glRasterPos2i(50, 50);
  char text[50];
  snprintf(text, sizeof(text), "Particles: %d", p_system->size());
  for (char *c = text; *c != '\0'; c++) {
    glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, *c); // Larger font
  }

  // Restore matrices for 3D rendering
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glEnable(GL_DEPTH_TEST);

  // Then continue with your 3D rendering code
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

  // Render upsampled particles
  glUniform1f(glGetUniformLocation(m_particles_program, "pointRadius"),
              upsampled_particle_radius);
  renderUpsampledParticles();

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
  case 'N':
    void one_step();
    one_step();
    break;
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

// thrust::fill(thrust::device, _buffer_num_surface_neighbors.addr(),
//              _buffer_num_surface_neighbors.addr() + num, 0);

// find_num_surface_neighbors<<<(num - 1) / block_size + 1, block_size>>>(
//     _buffer_num_surface_neighbors.addr(), particles->get_pos_ptr(),
//     particles->get_mass_ptr(), num, _cell_start_particle.addr(), _cell_size,
//     _cell_length, _density, _buffer_boundary.addr());
