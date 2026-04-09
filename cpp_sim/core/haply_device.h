#pragma once

#ifdef HAPTICS_ENABLED

#include <Eigen/Dense>
#include <hv/WebSocketClient.h>
#include <nlohmann/json.hpp>
#include <mutex>
#include <string>
#include <iostream>
#include <cmath>
#include <atomic>

using Vec3 = Eigen::Vector3d;
using json = nlohmann::json;

class HaplyDevice {
public:
    struct Config {
        std::string ws_url = "ws://localhost:10001";
        Vec3 workspace_center{0.0, -0.18, 0.15};
        Vec3 sim_offset{-0.017, 0.012, 0.000};
        double scale = 0.1;
        bool versagrip_correction = true;
        bool button_activates_knife = true;
        // Configurable axis mapping: pos_axes[i] = which raw axis (0=x,1=y,2=z) maps to sim axis i
        int pos_axes[3] = {0, 2, 1};      // default: sim.x=raw.x, sim.y=raw.z, sim.z=raw.y
        double pos_signs[3] = {1, 1, -1};  // default: flip Z
    };

    bool connect(const Config& cfg) {
        m_cfg = cfg;
        m_rot = Eigen::Matrix3d::Identity();

        reconn_setting_t reconn;
        reconn_setting_init(&reconn);
        reconn.min_delay = 1000;
        reconn.max_delay = 5000;
        reconn.delay_policy = 2;
        m_ws.setReconnect(&reconn);

        m_ws.onopen = [this]() {
            std::cout << "[Haply] WebSocket connected" << std::endl;
            m_connected = true;
        };

        m_ws.onclose = [this]() {
            std::cout << "[Haply] WebSocket disconnected" << std::endl;
            m_connected = false;
        };

        m_ws.onmessage = [this](const std::string& msg) {
            _on_message(msg);
        };

        m_ws.open(cfg.ws_url.c_str());

        // Wait briefly for connection
        for (int i = 0; i < 20 && !m_connected; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (m_connected) {
            std::cout << "[Haply] Connected to " << cfg.ws_url << std::endl;
        } else {
            std::cerr << "[Haply] Connection failed to " << cfg.ws_url << std::endl;
        }
        return m_connected;
    }

    void disconnect() {
        if (m_connected) {
            m_ws.close();
            m_connected = false;
        }
    }

    bool is_connected() const { return m_connected; }

    // Called from main thread each frame
    bool get_state(Vec3& out_pos, Eigen::Matrix3d& out_rot, bool& out_button) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_has_data) return false;
        out_pos = m_pos;
        out_rot = m_rot;
        out_button = m_button;
        return true;
    }

    // Phase 2: force feedback (no-op for now)
    void set_force(const Vec3& /*force_sim*/) {}

private:
    Config m_cfg;
    hv::WebSocketClient m_ws;
    std::atomic<bool> m_connected{false};

    std::mutex m_mutex;
    Vec3 m_pos{0, 0, 0};
    Eigen::Matrix3d m_rot = Eigen::Matrix3d::Identity();
    bool m_button = false;
    bool m_has_data = false;

    std::string m_inverse3_id;
    std::string m_versagrip_id;
    bool m_first_message = true;

    void _on_message(const std::string& msg) {
        json data;
        try {
            data = json::parse(msg);
        } catch (...) {
            return;
        }

        // First message: detect device IDs
        if (m_first_message) {
            m_first_message = false;
            if (data.contains("inverse3") && data["inverse3"].is_array() && !data["inverse3"].empty()) {
                m_inverse3_id = data["inverse3"][0]["device_id"].get<std::string>();
                std::cout << "[Haply] Inverse3 ID: " << m_inverse3_id << std::endl;
            }
            if (data.contains("verse_grip") && data["verse_grip"].is_array() && !data["verse_grip"].empty()) {
                m_versagrip_id = data["verse_grip"][0]["device_id"].get<std::string>();
                std::cout << "[Haply] VersaGrip ID: " << m_versagrip_id << std::endl;
            }
            if (m_inverse3_id.empty()) {
                std::cerr << "[Haply] No Inverse3 device found" << std::endl;
                return;
            }
        }

        // Parse Inverse3 position
        if (data.contains("inverse3") && !data["inverse3"].empty()) {
            const auto& inv = data["inverse3"][0];
            if (inv.contains("state") && inv["state"].contains("cursor_position")) {
                const auto& cp = inv["state"]["cursor_position"];
                double dx = cp["x"].get<double>();
                double dy = cp["y"].get<double>();
                double dz = cp["z"].get<double>();

                // Debug: print raw and mapped every 500ms
                {
                    static auto last_print = std::chrono::steady_clock::now();
                    auto now = std::chrono::steady_clock::now();
                    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print).count() > 500) {
                        last_print = now;
                        Vec3 mapped = _device_to_sim_pos(dx, dy, dz);
                        fprintf(stderr, "[Haply POS] raw=(%.4f, %.4f, %.4f) → sim=(%.5f, %.5f, %.5f)\n",
                                dx, dy, dz, mapped.x(), mapped.y(), mapped.z());
                    }
                }

                Vec3 pos = _device_to_sim_pos(dx, dy, dz);

                std::lock_guard<std::mutex> lock(m_mutex);
                m_pos = pos;
                m_has_data = true;
            }
        }

        // Parse VersaGrip orientation + button
        if (data.contains("verse_grip") && !data["verse_grip"].empty()) {
            const auto& vg = data["verse_grip"][0];
            if (vg.contains("state")) {
                const auto& state = vg["state"];
                if (state.contains("orientation")) {
                    double qx = state["orientation"]["x"].get<double>();
                    double qy = state["orientation"]["y"].get<double>();
                    double qz = state["orientation"]["z"].get<double>();
                    double qw = state["orientation"]["w"].get<double>();

                    // Debug: print raw quaternion every 500ms
                    {
                        static auto last_print = std::chrono::steady_clock::now();
                        auto now = std::chrono::steady_clock::now();
                        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print).count() > 500) {
                            last_print = now;
                            fprintf(stderr, "[Haply ROT] quat=(%.4f, %.4f, %.4f, %.4f)\n", qx, qy, qz, qw);
                        }
                    }

                    Eigen::Matrix3d rot = _device_to_sim_rot(qx, qy, qz, qw);

                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_rot = rot;
                }
                if (state.contains("button")) {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_button = state["button"].get<bool>();
                }
            }
        }

        // Send probe commands to keep data flowing
        _send_probe();
    }

    void _send_probe() {
        json request;

        if (!m_inverse3_id.empty()) {
            request["inverse3"] = json::array();
            request["inverse3"].push_back({
                {"device_id", m_inverse3_id},
                {"commands", {{"probe_position", json::object()}}}
            });
        }

        if (!m_versagrip_id.empty()) {
            request["verse_grip"] = json::array();
            request["verse_grip"].push_back({
                {"device_id", m_versagrip_id},
                {"commands", {{"probe_orientation", json::object()}}}
            });
        }

        m_ws.send(request.dump());
    }

    // Position: configurable axis mapping
    Vec3 _device_to_sim_pos(double dx, double dy, double dz) {
        // Subtract workspace center
        double centered[3] = {
            dx - m_cfg.workspace_center.x(),
            dy - m_cfg.workspace_center.y(),
            dz - m_cfg.workspace_center.z()
        };

        // Configurable axis permutation + sign
        double sim[3];
        for (int i = 0; i < 3; i++) {
            sim[i] = m_cfg.pos_signs[i] * centered[m_cfg.pos_axes[i]] * m_cfg.scale
                     + m_cfg.sim_offset[i];
        }
        return Vec3(sim[0], sim[1], sim[2]);
    }

    // Orientation: build permutation matrix from same axis config, then apply
    Eigen::Matrix3d _device_to_sim_rot(double qx, double qy, double qz, double qw) {
        Eigen::Quaterniond q_dev(qw, qx, qy, qz);
        Eigen::Matrix3d R_dev = q_dev.toRotationMatrix();

        // Handle correction (from chai3d CHaplyDevices.cpp lines 248-249)
        if (m_cfg.versagrip_correction) {
            Eigen::Matrix3d Rz_neg90 = Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            Eigen::Matrix3d Rz_pos90 = Eigen::AngleAxisd(+M_PI/2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            R_dev = Rz_pos90 * R_dev * Rz_neg90;
        }

        // Build permutation matrix P from axis config (same mapping as position)
        Eigen::Matrix3d P = Eigen::Matrix3d::Zero();
        for (int i = 0; i < 3; i++) {
            P(i, m_cfg.pos_axes[i]) = m_cfg.pos_signs[i];
        }

        // Apply as parent frame rotation: R_sim = P * R_dev
        return P * R_dev;
    }
};

#endif // HAPTICS_ENABLED
