#pragma once

#include "../core/particles.h"
#include "../core/cluster.h"
#include "../core/tet_mesh.h"
#include "../core/rigid_mesh.h"
#include "../solver/shape_matching_cpu.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <cmath>
#include <chrono>
#include <iostream>

class Viewer {
public:
    // 仿真器引用
    ShapeMatchingSolver* solver = nullptr;

    // Tet mesh（可选，如果提供则渲染表面 mesh 而不是点）
    TetMesh* tet_mesh = nullptr;
    PFCalculator* pf_calc = nullptr;
    CuttingManager* cutting_mgr = nullptr;  // 用损伤值染色
    bool draw_surface = true;   // 绘制表面 mesh
    bool draw_wireframe = true; // 绘制线框
    bool draw_points = false;   // 绘制粒子点

    // 相机设置（公开，可从外部调整）
    void set_camera(const Eigen::Vector3f& target, float distance,
                    float yaw = 0.3f, float pitch = 0.4f) {
        m_cam_target = target;
        m_cam_dist = distance;
        m_cam_yaw = yaw;
        m_cam_pitch = pitch;
    }

    // Step 0.5b: ask the main loop to exit after the current frame (used when a replay ends)
    void request_close() { if (m_window) glfwSetWindowShouldClose(m_window, GLFW_TRUE); }

    // 刀具状态
    Vec3 knife_pos = Vec3(0, 0.3, 0.09);
    Eigen::Matrix3d knife_rot = Eigen::Matrix3d::Identity();  // tool rotation
    // B (tool proxy): knife_pos/knife_rot are the INPUT target; the tool is drawn at the pose the
    // simulation used when draw_tool_pose is set
    bool draw_tool_pose = false;
    Vec3 tool_draw_pos = Vec3(0, 0, 0);
    Eigen::Matrix3d tool_draw_rot = Eigen::Matrix3d::Identity();
    double knife_radius = 0.03;
    bool knife_active = false;

    // 每帧跑多少个仿真 substep
    int substeps_per_frame = 4;

    // 回调：每帧仿真前调用（可选）
    std::function<void()> on_pre_step = nullptr;

    // GPU 模式：自定义 step 回调（替代 solver->step()）
    // 参数 = substeps 数量。回调负责执行仿真 + 同步粒子数据回 solver->particles
    std::function<void(int)> on_step = nullptr;

    // GPU 模式：外部 damage 数组（替代 cutting_mgr->get_particle_damage）
    std::vector<float>* particle_damage_array = nullptr;

    // Per-particle averaged α (mean of incident edge α). When non-null + size>0,
    // visualization overrides damage-based color with jet colormap on α value.
    // Used by Stage 1 validation experiment: per_edge_alpha=true → show α field.
    std::vector<float>* particle_alpha_array = nullptr;
    float particle_alpha_min = 1.0f;   // colormap range low
    float particle_alpha_max = 10.0f;  // colormap range high

    // Rigid mesh for rendering (optional)
    RigidMesh* rigid_mesh_render = nullptr;

    // Tool mesh for rendering (optional, moves with knife_pos)
    RigidMesh* tool_mesh_render = nullptr;

    // 当前活跃/移除粒子数（GPU 模式下由 on_step 回调更新）
    int num_active_particles = 0;
    int num_removed_particles = 0;
    int num_clusters_display = 0;  // 用于标题栏显示

    // Scheme B (cut_only mode): 外部 tet_broken 数组指针
    //   on_step 回调每 5 frame 更新 tet_broken_count。当 count 增加时,
    //   viewer 重建 surface_faces 暴露切口。
    //   nullptr → debulk 模式不启用,完全沿用旧逻辑(基于 removed 粒子重建)。
    const std::vector<int>* tet_broken_ptr = nullptr;
    int tet_broken_count = 0;
    int broken_edge_count = 0;     // render.split_tets: a new broken edge inside an already cut tet changes its pieces

    bool init(int width = 1280, int height = 720, const char* title = "P(F) Shape Matching") {
        m_width = width;
        m_height = height;

        if (!glfwInit()) {
            std::cerr << "Failed to init GLFW" << std::endl;
            return false;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 4);

        m_window = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!m_window) {
            std::cerr << "Failed to create window" << std::endl;
            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(m_window);
        glfwSwapInterval(0);  // 不限帧率
        glfwSetWindowUserPointer(m_window, this);

        // 回调
        glfwSetKeyCallback(m_window, key_callback);
        glfwSetMouseButtonCallback(m_window, mouse_button_callback);
        glfwSetCursorPosCallback(m_window, cursor_pos_callback);
        glfwSetScrollCallback(m_window, scroll_callback);
        glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);

        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) {
            std::cerr << "Failed to init GLEW" << std::endl;
            return false;
        }

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_MULTISAMPLE);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glClearColor(0.15f, 0.15f, 0.18f, 1.0f);

        _init_shaders();
        _init_buffers();
        _init_grid();
        _init_sphere();
        _init_surface();

        return true;
    }

    void run() {
        if (!solver) {
            std::cerr << "No solver attached!" << std::endl;
            return;
        }

        double last_time = glfwGetTime();
        int frame_count = 0;

        while (!glfwWindowShouldClose(m_window)) {
            glfwPollEvents();

            // FPS 计算
            double now = glfwGetTime();
            frame_count++;
            if (now - last_time >= 1.0) {
                char title[128];
                int disp_clusters = num_clusters_display > 0 ? num_clusters_display : (int)solver->clusters.size();
                snprintf(title, sizeof(title),
                    "P(F) Shape Matching%s | %d particles | %d clusters | %.1f FPS | %s%s",
                    on_step ? " [GPU]" : "",
                    solver->particles.size(), disp_clusters,
                    frame_count / (now - last_time),
                    m_paused ? "PAUSED" : "RUNNING",
                    m_move_mode ? (knife_active ? " | MOVE KNIFE (M)" : " | MOVE KNIFE (M) - knife OFF, press K") : "");
                glfwSetWindowTitle(m_window, title);
                frame_count = 0;
                last_time = now;
            }

            // 仿真步
            if (!m_paused) {
                if (on_pre_step) on_pre_step();

                if (on_step) {
                    // GPU 模式：使用自定义 step ��调
                    on_step(substeps_per_frame);

                    // 粒子被移除后重建表面（暴露内部新表面）
                    // cut_only 模式: 触发条件改为 tet_broken_count 增加(粒子永远不删)。
                    if (tet_mesh) {
                        int curr_removed = num_removed_particles;
                        int curr_broken = tet_broken_count;
                        static int prev_removed_gpu = 0;
                        static int prev_broken_gpu = 0;
                        static int prev_bedges_gpu = 0;
                        if (curr_removed > prev_removed_gpu || curr_broken > prev_broken_gpu ||
                            (tet_mesh->split_render && broken_edge_count > prev_bedges_gpu)) {
                            tet_mesh->rebuild_surface_faces(solver->particles, tet_broken_ptr);
                        }
                        prev_removed_gpu = curr_removed;
                        prev_broken_gpu = curr_broken;
                        prev_bedges_gpu = broken_edge_count;
                    }

                    // debug info
                    m_frame_count_debug++;
                    if (m_frame_count_debug <= 5 || m_frame_count_debug % 60 == 0) {
                        int N = solver->particles.size();
                        Vec3 cm = Vec3::Zero();
                        int active = 0;
                        for (int i = 0; i < N; i++) {
                            if (solver->particles[i].removed) continue;
                            cm += solver->particles[i].pos;
                            active++;
                        }
                        if (active > 0) cm /= active;
                        printf("[frame %4d] active=%d removed=%d clusters=%d | cm=(%.2f,%.2f,%.2f)\n",
                            m_frame_count_debug, active, N - active,
                            num_clusters_display, cm.x(), cm.y(), cm.z());
                    }
                } else {
                    // CPU 模式：原有逻辑
                    solver->knife_pos = knife_pos;
                    solver->knife_radius = knife_radius;
                    solver->knife_collision = knife_active;
                    int prev_removed = 0;
                    for (int i = 0; i < solver->particles.size(); i++) {
                        if (solver->particles[i].removed) prev_removed++;
                    }

                    for (int sub = 0; sub < substeps_per_frame; sub++) {
                        solver->step();
                    }

                    // 如果有新粒子被移除，重建表面 mesh
                    if (tet_mesh) {
                        int curr_removed = 0;
                        for (int i = 0; i < solver->particles.size(); i++) {
                            if (solver->particles[i].removed) curr_removed++;
                        }
                        if (curr_removed > prev_removed) {
                            tet_mesh->rebuild_surface_faces(solver->particles);
                        }
                    }

                    // 每 30 帧打印 debug info
                    m_frame_count_debug++;
                    if (m_frame_count_debug <= 5 || m_frame_count_debug % 60 == 0) {
                        int N = solver->particles.size();
                        Vec3 cm = Vec3::Zero();
                        int active_count = 0;
                        for (int i = 0; i < N; i++) {
                            if (solver->particles[i].removed) continue;
                            cm += solver->particles[i].pos;
                            active_count++;
                        }
                        if (active_count > 0) cm /= active_count;

                        int removed_count = N - active_count;
                        printf("[frame %4d] active=%d removed=%d clusters=%d | cm=(%.2f,%.2f,%.2f)\n",
                            m_frame_count_debug, active_count, removed_count,
                            (int)solver->clusters.size(), cm.x(), cm.y(), cm.z());
                    }
                }
            }

            // 渲染
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // 相机追踪粒子质心
            if (m_track_object && solver->particles.size() > 0) {
                Vec3 cm = Vec3::Zero();
                int N = solver->particles.size();
                for (int i = 0; i < N; i++) cm += solver->particles[i].pos;
                cm /= N;
                m_cam_target = Eigen::Vector3f((float)cm.x(), (float)cm.y(), (float)cm.z());
            }

            Eigen::Matrix4f view = _view_matrix();
            Eigen::Matrix4f proj = _projection_matrix();

            _draw_grid(view, proj);
            if (rigid_mesh_render) {
                _draw_rigid_mesh(view, proj);
            }
            if (tet_mesh) {
                _draw_surface(view, proj);
            }
            // 粒子点（按3键toggle）
            if (draw_points) {
                _draw_particles(view, proj);
            }
            // 刀具：mesh tool或sphere
            if (tool_mesh_render) {
                _draw_tool_mesh(view, proj);
            } else {
                _draw_knife(view, proj);
            }

            glfwSwapBuffers(m_window);
        }

        _cleanup();
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

private:
    GLFWwindow* m_window = nullptr;
    int m_width = 1280, m_height = 720;

    // 相机
    float m_cam_dist = 0.6f;
    float m_cam_yaw = 0.3f;
    float m_cam_pitch = 0.4f;
    Eigen::Vector3f m_cam_target = {0.09f, 0.15f, 0.09f};

    // 鼠标
    bool m_lmb_down = false, m_rmb_down = false;
    double m_last_mx = 0, m_last_my = 0;
    bool m_space_held = false;
    // Sticky knife-move mode (key M). Needed for remote desktops (e.g. TeamViewer) that deliver
    // every key as PRESS+RELEASE in the same instant, so "hold Space" can never be observed.
    bool m_move_mode = false;
    bool m_paused = false;
    bool m_track_object = true;  // 相机追踪物体质心

    // OpenGL 资源
    GLuint m_particle_vao = 0, m_particle_vbo = 0, m_particle_cbo = 0;
    GLuint m_grid_vao = 0, m_grid_vbo = 0;
    int m_grid_vert_count = 0;
    GLuint m_sphere_vao = 0, m_sphere_vbo = 0, m_sphere_ebo = 0;
    int m_sphere_idx_count = 0;
    GLuint m_surface_vao = 0, m_surface_vbo = 0, m_surface_nbo = 0, m_surface_ebo = 0;
    int m_surface_idx_count = 0;
    GLuint m_wireframe_vao = 0, m_wireframe_vbo = 0, m_wireframe_ebo = 0;
    int m_wireframe_idx_count = 0;
    GLuint m_rigid_vao = 0, m_rigid_vbo = 0, m_rigid_ibo = 0;
    int m_rigid_idx_count = 0;
    GLuint m_tool_vao = 0, m_tool_vbo = 0, m_tool_ibo = 0;
    int m_tool_idx_count = 0;
    GLuint m_point_shader = 0, m_line_shader = 0, m_mesh_shader = 0;
    int m_frame_count_debug = 0;

    // ---- Shaders ----
    void _init_shaders() {
        // 粒子 shader（带颜色）
        const char* point_vs = R"(
            #version 330 core
            layout(location=0) in vec3 aPos;
            layout(location=1) in vec3 aColor;
            uniform mat4 uMVP;
            uniform float uPointSize;
            out vec3 vColor;
            void main() {
                gl_Position = uMVP * vec4(aPos, 1.0);
                gl_PointSize = uPointSize;
                vColor = aColor;
            }
        )";
        const char* point_fs = R"(
            #version 330 core
            in vec3 vColor;
            out vec4 FragColor;
            void main() {
                // 圆形点
                vec2 c = gl_PointCoord * 2.0 - 1.0;
                if (dot(c,c) > 1.0) discard;
                float light = 0.7 + 0.3 * (1.0 - dot(c,c));
                FragColor = vec4(vColor * light, 1.0);
            }
        )";
        m_point_shader = _compile_program(point_vs, point_fs);

        // 线 shader（网格/wireframe）
        const char* line_vs = R"(
            #version 330 core
            layout(location=0) in vec3 aPos;
            uniform mat4 uMVP;
            void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
        )";
        const char* line_fs = R"(
            #version 330 core
            uniform vec4 uColor;
            out vec4 FragColor;
            void main() { FragColor = uColor; }
        )";
        m_line_shader = _compile_program(line_vs, line_fs);

        // Mesh shader（刀具球体用，简单 Blinn-Phong）
        const char* mesh_vs = R"(
            #version 330 core
            layout(location=0) in vec3 aPos;
            layout(location=1) in vec3 aNormal;
            uniform mat4 uMVP;
            uniform mat4 uModel;
            out vec3 vNormal;
            out vec3 vWorldPos;
            void main() {
                gl_Position = uMVP * vec4(aPos, 1.0);
                vWorldPos = vec3(uModel * vec4(aPos, 1.0));
                vNormal = mat3(transpose(inverse(uModel))) * aNormal;
            }
        )";
        const char* mesh_fs = R"(
            #version 330 core
            in vec3 vNormal;
            in vec3 vWorldPos;
            uniform vec4 uColor;
            uniform vec3 uLightDir;
            out vec4 FragColor;
            void main() {
                vec3 n = normalize(vNormal);
                float diff = max(dot(n, normalize(uLightDir)), 0.0) * 0.6 + 0.4;
                FragColor = vec4(uColor.rgb * diff, uColor.a);
            }
        )";
        m_mesh_shader = _compile_program(mesh_vs, mesh_fs);
    }

    GLuint _compile_shader(GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        int ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[512]; glGetShaderInfoLog(s, 512, nullptr, log);
            std::cerr << "Shader error: " << log << std::endl;
        }
        return s;
    }

    GLuint _compile_program(const char* vs_src, const char* fs_src) {
        GLuint vs = _compile_shader(GL_VERTEX_SHADER, vs_src);
        GLuint fs = _compile_shader(GL_FRAGMENT_SHADER, fs_src);
        GLuint prog = glCreateProgram();
        glAttachShader(prog, vs); glAttachShader(prog, fs);
        glLinkProgram(prog);
        glDeleteShader(vs); glDeleteShader(fs);
        return prog;
    }

    // ---- Buffers ----
    void _init_buffers() {
        glGenVertexArrays(1, &m_particle_vao);
        glGenBuffers(1, &m_particle_vbo);
        glGenBuffers(1, &m_particle_cbo);

        glBindVertexArray(m_particle_vao);
        // 位置 buffer
        glBindBuffer(GL_ARRAY_BUFFER, m_particle_vbo);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);
        // 颜色 buffer
        glBindBuffer(GL_ARRAY_BUFFER, m_particle_cbo);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);
    }

    // ---- 地板网格 ----
    void _init_grid() {
        std::vector<float> verts;
        float size = 0.5f;
        int lines = 20;
        for (int i = -lines; i <= lines; i++) {
            float t = (float)i / lines * size;
            // X 方向线
            verts.push_back(-size); verts.push_back(0); verts.push_back(t);
            verts.push_back(size);  verts.push_back(0); verts.push_back(t);
            // Z 方向线
            verts.push_back(t); verts.push_back(0); verts.push_back(-size);
            verts.push_back(t); verts.push_back(0); verts.push_back(size);
        }
        m_grid_vert_count = (int)verts.size() / 3;

        glGenVertexArrays(1, &m_grid_vao);
        glGenBuffers(1, &m_grid_vbo);
        glBindVertexArray(m_grid_vao);
        glBindBuffer(GL_ARRAY_BUFFER, m_grid_vbo);
        glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
    }

    // ---- 球体（刀具可视化） ----
    void _init_sphere() {
        std::vector<float> verts;
        std::vector<unsigned int> indices;
        int stacks = 16, slices = 24;

        for (int i = 0; i <= stacks; i++) {
            float phi = M_PI * i / stacks;
            for (int j = 0; j <= slices; j++) {
                float theta = 2.0f * M_PI * j / slices;
                float x = sin(phi) * cos(theta);
                float y = cos(phi);
                float z = sin(phi) * sin(theta);
                // position
                verts.push_back(x); verts.push_back(y); verts.push_back(z);
                // normal = same as position for unit sphere
                verts.push_back(x); verts.push_back(y); verts.push_back(z);
            }
        }

        for (int i = 0; i < stacks; i++) {
            for (int j = 0; j < slices; j++) {
                int a = i * (slices + 1) + j;
                int b = a + slices + 1;
                indices.push_back(a); indices.push_back(b); indices.push_back(a + 1);
                indices.push_back(a + 1); indices.push_back(b); indices.push_back(b + 1);
            }
        }
        m_sphere_idx_count = (int)indices.size();

        glGenVertexArrays(1, &m_sphere_vao);
        glGenBuffers(1, &m_sphere_vbo);
        glGenBuffers(1, &m_sphere_ebo);
        glBindVertexArray(m_sphere_vao);

        glBindBuffer(GL_ARRAY_BUFFER, m_sphere_vbo);
        glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_sphere_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
        glBindVertexArray(0);
    }

    // ---- 表面 mesh 初始化 ----
    void _init_surface() {
        // 表面 mesh VAO
        glGenVertexArrays(1, &m_surface_vao);
        glGenBuffers(1, &m_surface_vbo);
        glGenBuffers(1, &m_surface_nbo);
        glGenBuffers(1, &m_surface_ebo);

        // wireframe VAO
        glGenVertexArrays(1, &m_wireframe_vao);
        glGenBuffers(1, &m_wireframe_vbo);
        glGenBuffers(1, &m_wireframe_ebo);
    }

    void _upload_surface_mesh() {
        if (!tet_mesh || tet_mesh->num_surface_faces() == 0) return;

        // render.split_tets: vertices = particles + cut points (TetMesh::render_pos)
        const bool split = tet_mesh->split_render;
        int N = split ? tet_mesh->num_render_vertices() : (int)solver->particles.size();
        int F = tet_mesh->num_surface_faces();

        // 更新法线
        tet_mesh->update_normals(solver->particles);

        // 准备逐顶点数据
        std::vector<float> positions(N * 3);
        std::vector<float> normals(N * 3, 0.0f);

        for (int i = 0; i < N; i++) {
            const Vec3& p = split ? tet_mesh->render_pos[i] : solver->particles[i].pos;
            positions[i*3+0] = (float)p.x();
            positions[i*3+1] = (float)p.y();
            positions[i*3+2] = (float)p.z();
        }
        auto gone = [&](int v) { return solver->particles[split ? tet_mesh->render_owner(v) : v].removed; };

        // 逐面法线 → 逐顶点法线（累加后归一化）
        std::vector<int> normal_count(N, 0);
        for (int i = 0; i < F; i++) {
            const auto& f = tet_mesh->surface_faces[i];
            const Vec3& fn = tet_mesh->face_normals[i];
            for (int j = 0; j < 3; j++) {
                int vi = f[j];
                normals[vi*3+0] += (float)fn.x();
                normals[vi*3+1] += (float)fn.y();
                normals[vi*3+2] += (float)fn.z();
                normal_count[vi]++;
            }
        }
        for (int i = 0; i < N; i++) {
            if (normal_count[i] > 0) {
                float inv = 1.0f / normal_count[i];
                normals[i*3+0] *= inv;
                normals[i*3+1] *= inv;
                normals[i*3+2] *= inv;
                // normalize
                float len = std::sqrt(normals[i*3]*normals[i*3] + normals[i*3+1]*normals[i*3+1] + normals[i*3+2]*normals[i*3+2]);
                if (len > 1e-6f) { normals[i*3] /= len; normals[i*3+1] /= len; normals[i*3+2] /= len; }
            }
        }

        // 索引（跳过涉及 removed 粒子的三角面）
        std::vector<unsigned int> indices;
        indices.reserve(F * 3);
        for (int i = 0; i < F; i++) {
            int a = tet_mesh->surface_faces[i][0];
            int b = tet_mesh->surface_faces[i][1];
            int c = tet_mesh->surface_faces[i][2];
            // 任何一个顶点被移除 → 跳过这个面
            if (gone(a) || gone(b) || gone(c)) continue;
            indices.push_back(a);
            indices.push_back(b);
            indices.push_back(c);
        }
        m_surface_idx_count = (int)indices.size();

        // 上传表面 mesh
        glBindVertexArray(m_surface_vao);

        glBindBuffer(GL_ARRAY_BUFFER, m_surface_vbo);
        glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(float), positions.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, m_surface_nbo);
        glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_surface_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_DYNAMIC_DRAW);

        glBindVertexArray(0);

        // Wireframe（同样跳过 removed 粒子的边）
        std::unordered_set<int64_t> edge_set;
        std::vector<unsigned int> wire_indices;
        for (int i = 0; i < F; i++) {
            const auto& f = tet_mesh->surface_faces[i];
            if (gone(f[0]) || gone(f[1]) || gone(f[2])) continue;
            for (int j = 0; j < 3; j++) {
                int a = f[j], b = f[(j+1)%3];
                int64_t key = (int64_t)std::min(a,b) * 1000000LL + std::max(a,b);
                if (edge_set.insert(key).second) {
                    wire_indices.push_back(a);
                    wire_indices.push_back(b);
                }
            }
        }
        m_wireframe_idx_count = (int)wire_indices.size();

        glBindVertexArray(m_wireframe_vao);
        glBindBuffer(GL_ARRAY_BUFFER, m_surface_vbo);  // 共享位置 buffer
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_wireframe_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, wire_indices.size() * sizeof(unsigned int), wire_indices.data(), GL_DYNAMIC_DRAW);
        glBindVertexArray(0);
    }

    void _draw_surface(const Eigen::Matrix4f& view, const Eigen::Matrix4f& proj) {
        if (!tet_mesh) return;

        _upload_surface_mesh();
        if (m_surface_idx_count == 0) return;

        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f mvp = proj * view * model;

        // 计算相机位置用于光照
        float cx = m_cam_dist * cos(m_cam_pitch) * sin(m_cam_yaw);
        float cy = m_cam_dist * sin(m_cam_pitch);
        float cz = m_cam_dist * cos(m_cam_pitch) * cos(m_cam_yaw);
        Eigen::Vector3f cam_pos = m_cam_target + Eigen::Vector3f(cx, cy, cz);

        if (draw_surface) {
            glUseProgram(m_mesh_shader);
            glUniformMatrix4fv(glGetUniformLocation(m_mesh_shader, "uMVP"), 1, GL_FALSE, mvp.data());
            glUniformMatrix4fv(glGetUniformLocation(m_mesh_shader, "uModel"), 1, GL_FALSE, model.data());
            glUniform4f(glGetUniformLocation(m_mesh_shader, "uColor"), 0.85f, 0.35f, 0.3f, 1.0f);
            glUniform3f(glGetUniformLocation(m_mesh_shader, "uLightDir"), 0.3f, 1.0f, 0.5f);

            glBindVertexArray(m_surface_vao);
            glDrawElements(GL_TRIANGLES, m_surface_idx_count, GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);
        }

        if (draw_wireframe) {
            glUseProgram(m_line_shader);
            glUniformMatrix4fv(glGetUniformLocation(m_line_shader, "uMVP"), 1, GL_FALSE, mvp.data());
            glUniform4f(glGetUniformLocation(m_line_shader, "uColor"), 0.2f, 0.2f, 0.2f, 1.0f);

            // 轻微偏移防止 z-fighting
            glEnable(GL_POLYGON_OFFSET_LINE);
            glPolygonOffset(-1.0f, -1.0f);

            glBindVertexArray(m_wireframe_vao);
            glDrawElements(GL_LINES, m_wireframe_idx_count, GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);

            glDisable(GL_POLYGON_OFFSET_LINE);
        }
    }

    void _draw_rigid_mesh(const Eigen::Matrix4f& view, const Eigen::Matrix4f& proj) {
        if (!rigid_mesh_render || rigid_mesh_render->num_faces() == 0) return;

        // Upload rigid mesh data (once or if changed)
        if (m_rigid_vao == 0) {
            glGenVertexArrays(1, &m_rigid_vao);
            glGenBuffers(1, &m_rigid_vbo);
            glGenBuffers(1, &m_rigid_ibo);

            // Build vertex + normal data
            std::vector<float> vdata;
            std::vector<unsigned int> indices;
            for (int i = 0; i < rigid_mesh_render->num_faces(); i++) {
                auto& f = rigid_mesh_render->faces[i];
                auto& n = rigid_mesh_render->face_normals[i];
                for (int j = 0; j < 3; j++) {
                    auto& v = rigid_mesh_render->vertices[f[j]];
                    vdata.push_back((float)v.x());
                    vdata.push_back((float)v.y());
                    vdata.push_back((float)v.z());
                    vdata.push_back((float)n.x());
                    vdata.push_back((float)n.y());
                    vdata.push_back((float)n.z());
                    indices.push_back(i * 3 + j);
                }
            }
            m_rigid_idx_count = (int)indices.size();

            glBindVertexArray(m_rigid_vao);
            glBindBuffer(GL_ARRAY_BUFFER, m_rigid_vbo);
            glBufferData(GL_ARRAY_BUFFER, vdata.size() * sizeof(float), vdata.data(), GL_STATIC_DRAW);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_rigid_ibo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);
            glBindVertexArray(0);
        }

        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f mvp = proj * view * model;

        glUseProgram(m_mesh_shader);
        glUniformMatrix4fv(glGetUniformLocation(m_mesh_shader, "uMVP"), 1, GL_FALSE, mvp.data());
        glUniformMatrix4fv(glGetUniformLocation(m_mesh_shader, "uModel"), 1, GL_FALSE, model.data());
        glUniform4f(glGetUniformLocation(m_mesh_shader, "uColor"), 0.9f, 0.9f, 0.85f, 1.0f);  // bone color: off-white
        glUniform3f(glGetUniformLocation(m_mesh_shader, "uLightDir"), 0.3f, 1.0f, 0.5f);

        glBindVertexArray(m_rigid_vao);
        glDrawElements(GL_TRIANGLES, m_rigid_idx_count, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);
    }

    void _draw_tool_mesh(const Eigen::Matrix4f& view, const Eigen::Matrix4f& proj) {
        if (!tool_mesh_render || tool_mesh_render->num_faces() == 0) return;

        // Upload once
        if (m_tool_vao == 0) {
            glGenVertexArrays(1, &m_tool_vao);
            glGenBuffers(1, &m_tool_vbo);
            glGenBuffers(1, &m_tool_ibo);

            std::vector<float> vdata;
            std::vector<unsigned int> indices;
            for (int i = 0; i < tool_mesh_render->num_faces(); i++) {
                auto& f = tool_mesh_render->faces[i];
                auto& n = tool_mesh_render->face_normals[i];
                for (int j = 0; j < 3; j++) {
                    auto& v = tool_mesh_render->vertices[f[j]];
                    vdata.push_back((float)v.x());
                    vdata.push_back((float)v.y());
                    vdata.push_back((float)v.z());
                    vdata.push_back((float)n.x());
                    vdata.push_back((float)n.y());
                    vdata.push_back((float)n.z());
                    indices.push_back(i * 3 + j);
                }
            }
            m_tool_idx_count = (int)indices.size();

            glBindVertexArray(m_tool_vao);
            glBindBuffer(GL_ARRAY_BUFFER, m_tool_vbo);
            glBufferData(GL_ARRAY_BUFFER, vdata.size() * sizeof(float), vdata.data(), GL_STATIC_DRAW);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_tool_ibo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)(3*sizeof(float)));
            glEnableVertexAttribArray(1);
            glBindVertexArray(0);
        }

        // Model matrix: rotation + translation (tool mesh centered at origin)
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        // Fill rotation block
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                model(r, c) = (float)(draw_tool_pose ? tool_draw_rot(r, c) : knife_rot(r, c));
        // Translation
        const Vec3& tp = draw_tool_pose ? tool_draw_pos : knife_pos;
        model(0, 3) = (float)tp.x();
        model(1, 3) = (float)tp.y();
        model(2, 3) = (float)tp.z();

        Eigen::Matrix4f mvp = proj * view * model;

        glUseProgram(m_mesh_shader);
        glUniformMatrix4fv(glGetUniformLocation(m_mesh_shader, "uMVP"), 1, GL_FALSE, mvp.data());
        glUniformMatrix4fv(glGetUniformLocation(m_mesh_shader, "uModel"), 1, GL_FALSE, model.data());
        glUniform4f(glGetUniformLocation(m_mesh_shader, "uColor"), 0.7f, 0.7f, 0.75f, 1.0f);
        glUniform3f(glGetUniformLocation(m_mesh_shader, "uLightDir"), 0.3f, 1.0f, 0.5f);

        glBindVertexArray(m_tool_vao);
        glDrawElements(GL_TRIANGLES, m_tool_idx_count, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);
    }

    // ---- 绘制 ----
    void _draw_particles(const Eigen::Matrix4f& view, const Eigen::Matrix4f& proj) {
        int N = solver->particles.size();
        if (N == 0) return;

        // 准备位置和颜色数据（只渲染未移除的粒子）
        std::vector<float> positions;
        std::vector<float> colors;
        positions.reserve(N * 3);
        colors.reserve(N * 3);

        for (int i = 0; i < N; i++) {
            const auto& p = solver->particles[i];
            if (p.removed) continue;  // 跳过被移除的粒子

            positions.push_back((float)p.pos.x());
            positions.push_back((float)p.pos.y());
            positions.push_back((float)p.pos.z());

            // Priority 1: per-particle α (Stage 1 validation — show α field as jet colormap)
            if (particle_alpha_array && i < (int)particle_alpha_array->size()) {
                float a = (*particle_alpha_array)[i];
                float t = (a - particle_alpha_min) / std::max(particle_alpha_max - particle_alpha_min, 1e-6f);
                t = std::max(0.0f, std::min(1.0f, t));
                // Jet colormap: blue (low) → cyan → green → yellow → red (high)
                float r = std::max(0.0f, std::min(1.0f, 1.5f - 4.0f * std::abs(t - 0.75f)));
                float g = std::max(0.0f, std::min(1.0f, 1.5f - 4.0f * std::abs(t - 0.5f)));
                float b = std::max(0.0f, std::min(1.0f, 1.5f - 4.0f * std::abs(t - 0.25f)));
                colors.push_back(r);
                colors.push_back(g);
                colors.push_back(b);
            } else if (particle_damage_array && i < (int)particle_damage_array->size()) {
                float dmg = (*particle_damage_array)[i];
                colors.push_back(dmg);
                colors.push_back(0.15f);
                colors.push_back(1.0f - dmg);
            } else if (cutting_mgr) {
                float dmg = cutting_mgr->get_particle_damage(i);
                colors.push_back(dmg);
                colors.push_back(0.15f);
                colors.push_back(1.0f - dmg);
            } else {
                switch (p.layer_id) {
                    case 0: colors.push_back(0.9f); colors.push_back(0.25f); colors.push_back(0.2f); break;
                    case 1: colors.push_back(0.95f); colors.push_back(0.75f); colors.push_back(0.2f); break;
                    case 2: colors.push_back(0.2f); colors.push_back(0.5f); colors.push_back(0.9f); break;
                    default: colors.push_back(0.8f); colors.push_back(0.8f); colors.push_back(0.8f); break;
                }
            }
        }
        int render_count = (int)positions.size() / 3;

        // 上传到 GPU
        glBindBuffer(GL_ARRAY_BUFFER, m_particle_vbo);
        glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(float), positions.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, m_particle_cbo);
        glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float), colors.data(), GL_DYNAMIC_DRAW);

        // 绘制
        Eigen::Matrix4f mvp = proj * view;
        glUseProgram(m_point_shader);
        glUniformMatrix4fv(glGetUniformLocation(m_point_shader, "uMVP"), 1, GL_FALSE, mvp.data());
        glUniform1f(glGetUniformLocation(m_point_shader, "uPointSize"), 5.0f);

        glBindVertexArray(m_particle_vao);
        glDrawArrays(GL_POINTS, 0, render_count);
        glBindVertexArray(0);
    }

    void _draw_grid(const Eigen::Matrix4f& view, const Eigen::Matrix4f& proj) {
        Eigen::Matrix4f mvp = proj * view;
        glUseProgram(m_line_shader);
        glUniformMatrix4fv(glGetUniformLocation(m_line_shader, "uMVP"), 1, GL_FALSE, mvp.data());
        glUniform4f(glGetUniformLocation(m_line_shader, "uColor"), 0.3f, 0.3f, 0.3f, 1.0f);

        glBindVertexArray(m_grid_vao);
        glDrawArrays(GL_LINES, 0, m_grid_vert_count);
        glBindVertexArray(0);
    }

    void _draw_knife(const Eigen::Matrix4f& view, const Eigen::Matrix4f& proj) {
        // 构建 model 矩阵：平移 + 缩放
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        model(0, 3) = (float)knife_pos.x();
        model(1, 3) = (float)knife_pos.y();
        model(2, 3) = (float)knife_pos.z();
        model(0, 0) = model(1, 1) = model(2, 2) = (float)knife_radius;

        Eigen::Matrix4f mvp = proj * view * model;

        glUseProgram(m_mesh_shader);
        glUniformMatrix4fv(glGetUniformLocation(m_mesh_shader, "uMVP"), 1, GL_FALSE, mvp.data());
        glUniformMatrix4fv(glGetUniformLocation(m_mesh_shader, "uModel"), 1, GL_FALSE, model.data());
        glUniform4f(glGetUniformLocation(m_mesh_shader, "uColor"), 0.3f, 0.9f, 0.3f, 0.6f);
        glUniform3f(glGetUniformLocation(m_mesh_shader, "uLightDir"), 0.3f, 1.0f, 0.5f);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glBindVertexArray(m_sphere_vao);
        glDrawElements(GL_TRIANGLES, m_sphere_idx_count, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);

        glDisable(GL_BLEND);
    }

    // ---- 刀具碰撞 ----
    void _apply_knife_collision() {
        double max_push = solver->config.dt * 0.5;  // 每步最多推 0.5*dt 距离
        for (int i = 0; i < solver->particles.size(); i++) {
            auto& p = solver->particles[i];
            Vec3 diff = p.pos - knife_pos;
            double dist = diff.norm();
            if (dist < knife_radius && dist > 1e-10) {
                Vec3 n = diff / dist;
                double penetration = knife_radius - dist;
                double push = std::min(penetration * 0.1, max_push);
                p.pos += n * push;
                // 消除朝向刀具的速度分量
                double vn = p.vel.dot(n);
                if (vn < 0) p.vel -= vn * n;
            }
        }
    }

    // ---- 矩阵 ----
    Eigen::Matrix4f _view_matrix() {
        float cx = m_cam_dist * cos(m_cam_pitch) * sin(m_cam_yaw);
        float cy = m_cam_dist * sin(m_cam_pitch);
        float cz = m_cam_dist * cos(m_cam_pitch) * cos(m_cam_yaw);
        Eigen::Vector3f eye = m_cam_target + Eigen::Vector3f(cx, cy, cz);
        return _look_at(eye, m_cam_target, Eigen::Vector3f(0, 1, 0));
    }

    Eigen::Matrix4f _projection_matrix() {
        float aspect = (float)m_width / std::max(m_height, 1);
        return _perspective(45.0f * M_PI / 180.0f, aspect, 0.001f, 100.0f);
    }

    static Eigen::Matrix4f _look_at(const Eigen::Vector3f& eye, const Eigen::Vector3f& target, const Eigen::Vector3f& up) {
        Eigen::Vector3f f = (target - eye).normalized();
        Eigen::Vector3f r = f.cross(up).normalized();
        Eigen::Vector3f u = r.cross(f);
        Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
        m(0,0)=r.x(); m(0,1)=r.y(); m(0,2)=r.z(); m(0,3)=-r.dot(eye);
        m(1,0)=u.x(); m(1,1)=u.y(); m(1,2)=u.z(); m(1,3)=-u.dot(eye);
        m(2,0)=-f.x();m(2,1)=-f.y();m(2,2)=-f.z();m(2,3)=f.dot(eye);
        return m;
    }

    static Eigen::Matrix4f _perspective(float fov, float aspect, float near, float far) {
        float t = tan(fov / 2.0f);
        Eigen::Matrix4f m = Eigen::Matrix4f::Zero();
        m(0,0) = 1.0f / (aspect * t);
        m(1,1) = 1.0f / t;
        m(2,2) = -(far + near) / (far - near);
        m(2,3) = -2.0f * far * near / (far - near);
        m(3,2) = -1.0f;
        return m;
    }

    // ---- GLFW Callbacks ----
    // PFSM_INPUT_DEBUG=1 → log raw key/cursor events to stderr (diagnostics only)
    static bool input_debug() {
        static int on = -1;
        if (on < 0) { const char* v = std::getenv("PFSM_INPUT_DEBUG"); on = (v && *v && *v != '0') ? 1 : 0; }
        return on == 1;
    }
    static void key_callback(GLFWwindow* w, int key, int scancode, int action, int mods) {
        auto* self = (Viewer*)glfwGetWindowUserPointer(w);
        if (input_debug()) {
            const char* an = action == GLFW_PRESS ? "PRESS" : action == GLFW_RELEASE ? "RELEASE" : "REPEAT";
            const char* kn = glfwGetKeyName(key, scancode);
            fprintf(stderr, "[INPUT %.3f] key=%d(%s) scancode=%d %s mods=%d | space_held(before)=%d knife_active=%d\n",
                    glfwGetTime(), key, key == GLFW_KEY_SPACE ? "SPACE" : (kn ? kn : "?"), scancode, an, mods,
                    (int)self->m_space_held, (int)self->knife_active);
        }
        if (key == GLFW_KEY_SPACE) {
            self->m_space_held = (action != GLFW_RELEASE);
        }
        if (action == GLFW_PRESS) {
            if (key == GLFW_KEY_P) self->m_paused = !self->m_paused;
            if (key == GLFW_KEY_K) self->knife_active = !self->knife_active;
            if (key == GLFW_KEY_M) {
                self->m_move_mode = !self->m_move_mode;
                printf("[input] knife move mode %s (mouse = %s)\n", self->m_move_mode ? "ON" : "OFF",
                       self->m_move_mode ? "move knife, scroll = forward/back" : "camera");
            }
            if (key == GLFW_KEY_R && !(mods & GLFW_MOD_SHIFT)) {
                self->knife_pos = Vec3(0.09, 0.3, 0.09);
                self->knife_rot = Eigen::Matrix3d::Identity();
            }
            if (key == GLFW_KEY_1) self->draw_surface = !self->draw_surface;
            if (key == GLFW_KEY_2) self->draw_wireframe = !self->draw_wireframe;
            if (key == GLFW_KEY_3) self->draw_points = !self->draw_points;
            if (key == GLFW_KEY_T) self->m_track_object = !self->m_track_object;
            if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, GLFW_TRUE);
        }
        // Tool rotation: X/Y/Z keys (body frame, hold to rotate, Shift for reverse)
        if (action == GLFW_PRESS || action == GLFW_REPEAT) {
            double angle = (mods & GLFW_MOD_SHIFT) ? -0.05 : 0.05;  // ~3 degrees
            Eigen::Matrix3d dR;
            if (key == GLFW_KEY_X) {
                dR = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()).toRotationMatrix();
                self->knife_rot = self->knife_rot * dR;  // 右乘 = body frame
            }
            if (key == GLFW_KEY_Y) {
                dR = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()).toRotationMatrix();
                self->knife_rot = self->knife_rot * dR;
            }
            if (key == GLFW_KEY_Z) {
                dR = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
                self->knife_rot = self->knife_rot * dR;
            }
        }
    }

    static void mouse_button_callback(GLFWwindow* w, int button, int action, int /*mods*/) {
        auto* self = (Viewer*)glfwGetWindowUserPointer(w);
        if (button == GLFW_MOUSE_BUTTON_LEFT)  self->m_lmb_down = (action == GLFW_PRESS);
        if (button == GLFW_MOUSE_BUTTON_RIGHT) self->m_rmb_down = (action == GLFW_PRESS);
        glfwGetCursorPos(w, &self->m_last_mx, &self->m_last_my);
    }

    // 从相机角度计算 right/up/forward 向量
    void _get_camera_axes(Eigen::Vector3f& right, Eigen::Vector3f& up, Eigen::Vector3f& fwd) const {
        float cp = cos(m_cam_pitch), sp = sin(m_cam_pitch);
        float cy = cos(m_cam_yaw), sy = sin(m_cam_yaw);
        fwd = Eigen::Vector3f(-cp * sy, -sp, -cp * cy);
        right = fwd.cross(Eigen::Vector3f(0, 1, 0)).normalized();
        up = right.cross(fwd).normalized();
    }

    static void cursor_pos_callback(GLFWwindow* w, double x, double y) {
        auto* self = (Viewer*)glfwGetWindowUserPointer(w);
        double dx = x - self->m_last_mx;
        double dy = y - self->m_last_my;
        self->m_last_mx = x;
        self->m_last_my = y;

        // 限制鼠标最大 delta（参考 xpbd-tissue-sim PushingSimulation）
        const double max_delta = 20.0;
        dx = std::max(-max_delta, std::min(max_delta, dx));
        dy = std::max(-max_delta, std::min(max_delta, dy));

        Eigen::Vector3f right, up, fwd;
        self->_get_camera_axes(right, up, fwd);

        if (input_debug()) {
            static double last_t = -1.0;
            double now = glfwGetTime();
            if (now - last_t > 0.1 || self->m_space_held) {   // throttle when nothing interesting
                fprintf(stderr, "[INPUT %.3f] cursor dx=%.1f dy=%.1f | space_held=%d knife_active=%d lmb=%d rmb=%d\n",
                        now, dx, dy, (int)self->m_space_held, (int)self->knife_active,
                        (int)self->m_lmb_down, (int)self->m_rmb_down);
                last_t = now;
            }
        }
        if ((self->m_space_held || self->m_move_mode) && self->knife_active) {
            // Space(按住) 或 M(切换) + 移动鼠标：移动刀具（灵敏度和相机距离成比例）
            float scale = self->m_cam_dist * 0.002f;

            self->knife_pos += Vec3(right.x(), right.y(), right.z()) * dx * scale;
            self->knife_pos -= Vec3(up.x(), up.y(), up.z()) * dy * scale;
        } else if (self->m_lmb_down) {
            // LMB: 旋转相机
            self->m_cam_yaw += (float)dx * 0.005f;
            self->m_cam_pitch += (float)dy * 0.005f;
            self->m_cam_pitch = std::max(-1.5f, std::min(1.5f, self->m_cam_pitch));
        } else if (self->m_rmb_down) {
            // RMB: 平移相机
            float scale = self->m_cam_dist * 0.003f;
            self->m_cam_target += right * (float)dx * scale;
            self->m_cam_target -= up * (float)dy * scale;
        }
    }

    static void scroll_callback(GLFWwindow* w, double /*dx*/, double dy) {
        auto* self = (Viewer*)glfwGetWindowUserPointer(w);

        // 限制滚轮 delta
        double limited_dy = std::max(-2.0, std::min(2.0, dy));

        Eigen::Vector3f right, up, fwd;
        self->_get_camera_axes(right, up, fwd);

        if ((self->m_space_held || self->m_move_mode) && self->knife_active) {
            // Space(按住) 或 M(切换) + Scroll：刀具沿视线方向前后移动
            float scale = self->m_cam_dist * 0.1f;
            self->knife_pos += Vec3(fwd.x(), fwd.y(), fwd.z()) * limited_dy * scale;
        } else {
            // Scroll: 缩放相机
            self->m_cam_dist *= (1.0f - (float)limited_dy * 0.1f);
            self->m_cam_dist = std::max(0.05f, std::min(10.0f, self->m_cam_dist));
        }
    }

    static void framebuffer_size_callback(GLFWwindow* w, int width, int height) {
        auto* self = (Viewer*)glfwGetWindowUserPointer(w);
        self->m_width = width;
        self->m_height = height;
        glViewport(0, 0, width, height);
    }

    void _cleanup() {
        glDeleteVertexArrays(1, &m_particle_vao);
        glDeleteBuffers(1, &m_particle_vbo);
        glDeleteBuffers(1, &m_particle_cbo);
        glDeleteVertexArrays(1, &m_grid_vao);
        glDeleteBuffers(1, &m_grid_vbo);
        glDeleteVertexArrays(1, &m_sphere_vao);
        glDeleteBuffers(1, &m_sphere_vbo);
        glDeleteBuffers(1, &m_sphere_ebo);
        glDeleteVertexArrays(1, &m_surface_vao);
        glDeleteBuffers(1, &m_surface_vbo);
        glDeleteBuffers(1, &m_surface_nbo);
        glDeleteBuffers(1, &m_surface_ebo);
        glDeleteVertexArrays(1, &m_wireframe_vao);
        glDeleteBuffers(1, &m_wireframe_vbo);
        glDeleteBuffers(1, &m_wireframe_ebo);
        glDeleteProgram(m_point_shader);
        glDeleteProgram(m_line_shader);
        glDeleteProgram(m_mesh_shader);
    }
};
