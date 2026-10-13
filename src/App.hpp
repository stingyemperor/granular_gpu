#pragma once
#include "Renderer.hpp"
#include "SceneBuilder.hpp"
#include "SceneConfig.hpp"
#include <string>
#include <vector>

// Owns the simulation, the renderer and everything the UI can change.
// Fields are public so the ImGui panel (Ui.cu) can edit them directly.
class App {
public:
  explicit App(const std::string &scene_path);
  App(const App &) = delete;
  App &operator=(const App &) = delete;

  // GL setup that needs a window; call once after glewInit().
  void initGL();
  // Saves stats and frees GPU resources; call before exiting.
  void shutdown();

  // Loads a scene file and rebuilds the simulation. On a bad file the current
  // scene keeps running and the error is shown in the UI.
  bool loadScene(std::string path);
  // Rebuilds the simulation from `config`, applying any edits made in the UI.
  void reset();
  void step();
  void setRunning(bool run);
  void explode();
  void saveStats();
  void quit();

  // GLUT callbacks (forwarded from main.cu)
  void display();
  void keyboard(unsigned char key);
  void special(int key);
  void mouse(int button, int state, int x, int y);
  void motion(int x, int y);
  void wheel(int dir);

  SceneConfig config;
  BuiltScene scene;
  std::vector<std::string> scene_files; // scenes/*.json for the picker

  Camera camera;
  RenderOptions render;
  bool show_panel = true;

  int frame = 0;
  bool running = false;
  bool animate = false; // run the scripted obj animation (excavator)
  float explosion_force = 50.0f;
  std::vector<float> recent_step_ms; // ring buffer for the timing plot
  int recent_step_index = 0;
  std::string status; // last error or info message for the UI

private:
  void build(const SceneConfig &new_config);

  Renderer _renderer;
  bool _fullscreen = false;
  bool _dragging = false;
  int _last_mouse[2] = {0, 0};
};
