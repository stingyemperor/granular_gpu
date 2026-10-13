#include "App.hpp"
#include "Ui.hpp"
#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"
#include <algorithm>
#include <filesystem>
#include <numeric>
#include <string>

namespace {

void help(const char *text) {
  ImGui::SameLine();
  ImGui::TextDisabled("(?)");
  if (ImGui::BeginItemTooltip()) {
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.0f);
    ImGui::TextUnformatted(text);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

std::string stem(const std::string &path) {
  return std::filesystem::path(path).stem().string();
}

void simulationSection(App &app) {
  if (!ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
    return;

  const std::string current = stem(app.config.path);
  if (ImGui::BeginCombo("Scene", current.c_str())) {
    for (const auto &file : app.scene_files) {
      const std::string name = stem(file);
      if (ImGui::Selectable(name.c_str(), name == current)) {
        app.loadScene(file);
      }
    }
    ImGui::EndCombo();
  }

  const float button_w =
      ImGui::GetContentRegionAvail().x / 3.0f - ImGui::GetStyle().ItemSpacing.x;
  if (ImGui::Button(app.running ? "Pause" : "Play", ImVec2(button_w, 0)))
    app.setRunning(!app.running);
  ImGui::SameLine();
  ImGui::BeginDisabled(app.running);
  if (ImGui::Button("Step", ImVec2(button_w, 0)))
    app.step();
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Reset", ImVec2(button_w, 0)))
    app.reset();

  if (ImGui::TreeNode("Parameters")) {
    ImGui::TextDisabled("Applied on Reset");
    auto &c = app.config;
    ImGui::InputFloat("dt", &c.dt, 0.0001f, 0.001f, "%.4f");
    c.dt = std::max(c.dt, 1e-5f);
    ImGui::InputFloat3("Gravity", &c.gravity.x, "%.2f");
    ImGui::Checkbox("Adaptive sampling", &c.adaptive);
    help("Merge interior particles into heavier ones and split them again "
         "near the surface.");
    ImGui::InputInt3("Particle grid", &c.particle_grid.x);
    c.particle_grid = make_int3(std::max(c.particle_grid.x, 1),
                                std::max(c.particle_grid.y, 1),
                                std::max(c.particle_grid.z, 1));
    ImGui::InputInt("Upsample count", &c.upsample_count);
    c.upsample_count = std::clamp(c.upsample_count, 0, 200);
    if (ImGui::Button("Reload scene file"))
      app.loadScene(c.path);
    ImGui::TreePop();
  }
}

void statsSection(App &app) {
  if (!ImGui::CollapsingHeader("Statistics", ImGuiTreeNodeFlags_DefaultOpen))
    return;

  const auto &sys = *app.scene.system;
  ImGui::Text("Frame          %d  (t = %.3f s)", app.frame,
              app.frame * app.config.dt);
  ImGui::Text("Particles      %d", sys.size());
  ImGui::Text("Upsampled      %d", sys.upsampled_size());
  ImGui::Text("Boundary       %d", sys.boundary_size());

  const auto &stats = app.scene.system->get_solver().get_particle_stats();
  if (!stats.empty()) {
    const auto &last = stats.back();
    ImGui::Text("Last step      %d merged, %d split", std::get<0>(last),
                std::get<1>(last));
  }

  const auto &samples = app.recent_step_ms;
  const float latest =
      samples[(app.recent_step_index + samples.size() - 1) % samples.size()];
  const float peak = *std::max_element(samples.begin(), samples.end());
  char overlay[64];
  snprintf(overlay, sizeof(overlay), "%.2f ms/step", latest);
  ImGui::PlotLines("##steptime", samples.data(), samples.size(),
                   app.recent_step_index, overlay, 0.0f,
                   std::max(peak * 1.2f, 1.0f),
                   ImVec2(-FLT_MIN, 50.0f * ImGui::GetIO().FontGlobalScale));
}

void renderSection(App &app) {
  if (!ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen))
    return;

  auto &r = app.render;
  ImGui::Checkbox("Particles", &r.show_particles);
  ImGui::SameLine();
  ImGui::Checkbox("Upsampled", &r.show_upsampled);
  ImGui::Checkbox("Boundary", &r.show_boundary);
  ImGui::SameLine();
  ImGui::Checkbox("Domain box", &r.show_domain);

  const char *modes[] = {"Surface", "Distance to surface", "Mass"};
  ImGui::Combo("Color by", &r.color_mode, modes, IM_ARRAYSIZE(modes));
  help("Coloring of the coarse simulation particles. Mass runs from blue "
       "(light) to red (heavy).");
  ImGui::SliderFloat("Point size", &r.point_scale, 0.1f, 4.0f, "%.2fx",
                     ImGuiSliderFlags_Logarithmic);
  ImGui::ColorEdit3("Background", r.background);
  if (ImGui::Button("Reset camera"))
    app.camera.reset();
}

void interactionSection(App &app) {
  if (!ImGui::CollapsingHeader("Interaction"))
    return;

  ImGui::SliderFloat("Force", &app.explosion_force, 1.0f, 500.0f, "%.0f",
                     ImGuiSliderFlags_Logarithmic);
  ImGui::SameLine();
  if (ImGui::Button("Explode"))
    app.explode();
  help("Pushes all particles away from their center of mass (key E).");

  if (app.config.scene == SceneType::Excavator && app.scene.has_obj) {
    ImGui::Checkbox("Animate bucket", &app.animate);
    help("Moves the bucket sideways, then tilts it by 45 degrees. Reset to "
         "start over.");
  }
}

void exportSection(App &app) {
  if (!ImGui::CollapsingHeader("Output"))
    return;

  auto &out = app.config.output;
  ImGui::TextWrapped("Directory: %s", out.dir.c_str());

  ImGui::SeparatorText("Every frame");
  auto &x = out.exports;
  ImGui::Checkbox("Upsampled (VTK)", &x.upsampled);
  ImGui::Checkbox("Upsampled, centered (VTK)", &x.upsampled_centered);
  help("Moved so the centroid is at the origin and the lowest particle at "
       "height 0.");
  ImGui::Checkbox("Upsampled (CSV)", &x.upsampled_csv);
  ImGui::Checkbox("Particles by mass (VTK)", &x.by_mass);
  if (app.scene.has_obj) {
    ImGui::Checkbox("Obj boundary (VTK)", &x.animated_boundary);
    ImGui::Checkbox("Obj boundary, centered (VTK)",
                    &x.animated_boundary_centered);
  }
  ImGui::InputInt("Start frame", &x.start_frame);

  ImGui::SeparatorText("On pause / exit");
  ImGui::Checkbox("stats.csv", &out.stats_csv);
  ImGui::SameLine();
  ImGui::Checkbox("frame_times.txt", &out.frame_times);
  if (ImGui::Button("Save now"))
    app.saveStats();
}

void controlsSection() {
  if (!ImGui::CollapsingHeader("Controls"))
    return;
  ImGui::TextUnformatted("Space      play / pause\n"
                         "N          single step\n"
                         "E          explode\n"
                         "P          toggle particles\n"
                         "Drag       rotate view\n"
                         "Wheel , .  zoom\n"
                         "R          reset camera\n"
                         "H          hide this panel\n"
                         "F11        fullscreen\n"
                         "Q / Esc    quit");
}

} // namespace

void drawUi(App &app) {
  ImGui_ImplOpenGL2_NewFrame();
  ImGui_ImplGLUT_NewFrame();
  ImGui::NewFrame();

  const float scale = ImGui::GetIO().FontGlobalScale;
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSizeConstraints(ImVec2(380 * scale, 0),
                                      ImVec2(380 * scale, FLT_MAX));
  ImGui::Begin("Granular GPU", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

  if (!app.status.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
    ImGui::TextWrapped("%s", app.status.c_str());
    ImGui::PopStyleColor();
    if (ImGui::SmallButton("Dismiss"))
      app.status.clear();
  }

  simulationSection(app);
  statsSection(app);
  renderSection(app);
  interactionSection(app);
  exportSection(app);
  controlsSection();

  ImGui::End();
  ImGui::Render();
  ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
}
