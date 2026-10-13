#include "App.hpp"
#include "Export.hpp"
#include "Ui.hpp"
#include <GL/freeglut.h>
#include <algorithm>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace {

const int kTimingSamples = 120;

std::vector<std::string> listSceneFiles() {
  std::vector<std::string> files;
  const fs::path dir = fs::path(GRANULAR_SOURCE_DIR) / "scenes";
  if (fs::is_directory(dir)) {
    for (const auto &entry : fs::directory_iterator(dir)) {
      if (entry.path().extension() == ".json") {
        files.push_back(entry.path().string());
      }
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

} // namespace

App::App(const std::string &scene_path)
    : config(loadSceneConfig(scene_path)), scene_files(listSceneFiles()),
      recent_step_ms(kTimingSamples, 0.0f) {}

void App::initGL() {
  _renderer.init();
  build(config);
}

void App::shutdown() {
  if (scene.system) {
    saveStats();
  }
  _renderer.release();
  scene = BuiltScene();
}

void App::build(const SceneConfig &new_config) {
  // free the old system first; both may not fit in GPU memory at once
  scene = BuiltScene();
  config = new_config;
  scene = buildScene(config);
  animate = config.animate_obj;
  frame = 0;
  std::fill(recent_step_ms.begin(), recent_step_ms.end(), 0.0f);
  std::cout << "Loaded " << config.path << " (" << sceneTypeName(config.scene)
            << "): " << scene.system->size() << " particles, "
            << scene.system->upsampled_size() << " upsampled, "
            << scene.system->boundary_size() << " boundary" << std::endl;
}

bool App::loadScene(std::string path) {
  SceneConfig new_config;
  try {
    new_config = loadSceneConfig(path);
  } catch (const std::exception &e) {
    status = e.what();
    std::cerr << "Error loading scene: " << status << std::endl;
    return false;
  }
  if (scene.system) {
    saveStats();
  }
  build(new_config);
  status.clear();
  return true;
}

void App::reset() {
  if (scene.system) {
    saveStats();
  }
  build(config);
}

void App::step() {
  if (animate) {
    updateAnimation(scene.animation, *scene.system->get_boundaries(),
                    config.dt);
  }
  const float ms = scene.system->step();
  recent_step_ms[recent_step_index] = ms;
  recent_step_index = (recent_step_index + 1) % kTimingSamples;
  ++frame;
  exportFrame(config, *scene.system, frame);
}

void App::setRunning(bool run) {
  if (running && !run) {
    saveStats();
  }
  running = run;
}

void App::explode() {
  auto particles = scene.system->get_particles_non_const();
  scene.system->get_solver().trigger_explosion(particles, explosion_force);
}

void App::saveStats() { saveRunStats(config, *scene.system); }

void App::quit() { glutLeaveMainLoop(); }

void App::display() {
  if (running) {
    step();
  }

  _renderer.draw(*scene.system, config, camera, render,
                 glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
  if (show_panel) {
    drawUi(*this);
  }

  glutSwapBuffers();
  glutPostRedisplay();
}

void App::keyboard(unsigned char key) {
  switch (key) {
  case ' ':
    setRunning(!running);
    break;
  case 'n':
  case 'N':
    step();
    break;
  case ',':
    camera.zoom *= 1.2f;
    break;
  case '.':
    camera.zoom /= 1.2f;
    break;
  case 'r':
  case 'R':
    camera.reset();
    break;
  case 'p':
  case 'P':
    render.show_particles = !render.show_particles;
    break;
  case 'e':
  case 'E':
    explode();
    break;
  case 'h':
  case 'H':
    show_panel = !show_panel;
    break;
  case 'q':
  case 'Q':
  case 27: // Esc
    quit();
    break;
  }
}

void App::special(int key) {
  if (key == GLUT_KEY_F11) {
    _fullscreen = !_fullscreen;
    if (_fullscreen) {
      glutFullScreen();
    } else {
      glutLeaveFullScreen();
    }
  }
}

void App::mouse(int button, int state, int x, int y) {
  if (button == GLUT_LEFT_BUTTON) {
    _dragging = state == GLUT_DOWN;
    _last_mouse[0] = x;
    _last_mouse[1] = y;
  }
}

void App::motion(int x, int y) {
  if (_dragging) {
    camera.rot_x += (y - _last_mouse[1]) * 0.25f;
    camera.rot_y += (x - _last_mouse[0]) * 0.25f;
  }
  _last_mouse[0] = x;
  _last_mouse[1] = y;
}

void App::wheel(int dir) { camera.zoom *= dir > 0 ? 1.1f : 1.0f / 1.1f; }
