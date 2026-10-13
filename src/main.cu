#include <GL/glew.h>

#include "App.hpp"
#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <memory>

namespace fs = std::filesystem;

namespace {

std::unique_ptr<App> app;

void printUsage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " [scene]\n\n"
            << "  scene  path to a scene .json file, or the name of one in\n"
            << "         scenes/ (e.g. \"funnel\"). Defaults to box.\n";
}

// Accepts a path, a path relative to the project root, or a scene name.
std::string findScene(const std::string &arg) {
  const fs::path root(GRANULAR_SOURCE_DIR);
  for (const fs::path &candidate :
       {fs::path(arg), root / arg, root / "scenes" / (arg + ".json")}) {
    if (fs::is_regular_file(candidate)) {
      return candidate.string();
    }
  }
  return arg; // let the loader report the error
}

void display() { app->display(); }

// Called by freeglut while the GL context still exists, both when the window
// is closed and when the main loop exits.
void onClose() {
  if (!app) {
    return;
  }
  app->shutdown();
  app.reset();
  ImGui_ImplOpenGL2_Shutdown();
  ImGui_ImplGLUT_Shutdown();
  ImGui::DestroyContext();
}

void reshape(int w, int h) { ImGui_ImplGLUT_ReshapeFunc(w, h); }

void keyboard(unsigned char c, int x, int y) {
  ImGui_ImplGLUT_KeyboardFunc(c, x, y);
  if (!ImGui::GetIO().WantCaptureKeyboard) {
    app->keyboard(c);
  }
}

void special(int key, int x, int y) {
  ImGui_ImplGLUT_SpecialFunc(key, x, y);
  if (!ImGui::GetIO().WantCaptureKeyboard) {
    app->special(key);
  }
}

void mouse(int button, int state, int x, int y) {
  ImGui_ImplGLUT_MouseFunc(button, state, x, y);
  // always forward releases so a drag that ends over the panel stops
  if (!ImGui::GetIO().WantCaptureMouse || state == GLUT_UP) {
    app->mouse(button, state, x, y);
  }
}

void motion(int x, int y) {
  ImGui_ImplGLUT_MotionFunc(x, y);
  app->motion(x, y);
}

void wheel(int button, int dir, int x, int y) {
  ImGui_ImplGLUT_MouseWheelFunc(button, dir, x, y);
  if (!ImGui::GetIO().WantCaptureMouse) {
    app->wheel(dir);
  }
}

void initImGui() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.IniFilename = nullptr; // don't write imgui.ini into the working dir

  ImGui::StyleColorsDark();
  // scale up the UI on high-resolution screens
  const float scale = std::max(1.0f, glutGet(GLUT_SCREEN_HEIGHT) / 1080.0f);
  ImGui::GetStyle().ScaleAllSizes(scale);
  io.FontGlobalScale = scale;

  ImGui_ImplGLUT_Init();
  ImGui_ImplOpenGL2_Init();
}

} // namespace

int main(int argc, char *argv[]) {
  glutInit(&argc, argv);

  std::string scene_arg = "box";
  if (argc > 1) {
    const std::string arg = argv[1];
    if (arg == "-h" || arg == "--help") {
      printUsage(argv[0]);
      return 0;
    }
    scene_arg = arg;
  }

  try {
    app = std::make_unique<App>(findScene(scene_arg));
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n\n";
    printUsage(argv[0]);
    return 1;
  }

  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
  glutInitWindowSize(1600, 1000);
  glutCreateWindow("Granular GPU");
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

  if (glewInit() != GLEW_OK) {
    std::cerr << "Error: failed to initialize GLEW" << std::endl;
    return 1;
  }

  initImGui();
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutKeyboardFunc(keyboard);
  glutKeyboardUpFunc(ImGui_ImplGLUT_KeyboardUpFunc);
  glutSpecialFunc(special);
  glutSpecialUpFunc(ImGui_ImplGLUT_SpecialUpFunc);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutPassiveMotionFunc(ImGui_ImplGLUT_MotionFunc);
  glutMouseWheelFunc(wheel);
  glutCloseFunc(onClose);

  int status = 0;
  try {
    app->initGL();
    glutMainLoop();
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    status = 1;
  }

  onClose(); // no-op if freeglut already called it
  cudaDeviceReset();
  return status;
}
