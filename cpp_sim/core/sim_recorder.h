#pragma once
// Step 0.5a — trajectory recording + per-frame statistics for the interactive viewer.
//
// TrajectoryRecorder (--record <file>): one line per SIMULATED frame with exactly what the
//   GPU solver received for the tool (position, row-major local->world rotation, active flag)
//   plus the wall-clock time. Recording starts at frame 0 (program start, incl. settling) so a
//   replay can start from the rest state. Paused frames do not advance the sim and are not
//   written. The header carries config/mesh hashes so a replay can refuse a changed setup.
//
// FrameStatsLogger (--stats <file>): one line per frame AFTER the full viewer pipeline
//   (GPU step, download, orphan cleanup, BFS split), i.e. what the user actually sees.
//   Includes scale-free "artifact" metrics: particles moving more than one mean particle
//   spacing in a single frame ("fast"), and particles leaving the initial bounding box
//   expanded by 10% of its extent ("out").
//
// Pure observation: nothing here writes back into the simulation.

#include <cuda_runtime.h>   // float3
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace simrec {

inline uint64_t fnv1a_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return 0;
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    uint64_t h = 1469598103934665603ULL;
    for (char c : buf) { h ^= (unsigned char)c; h *= 1099511628211ULL; }
    return h;
}

struct RecordMeta {
    std::string config_path, mesh_path, tool_mesh_path;
    float dt = 0, knife_radius = 0;
    int substeps = 0, tool_type = 0, num_iterations = 0, N = 0;
};

class TrajectoryRecorder {
public:
    bool open(const std::string& path, const RecordMeta& m) {
        f_ = std::fopen(path.c_str(), "w");
        if (!f_) { std::fprintf(stderr, "[record] cannot open %s\n", path.c_str()); return false; }
        std::time_t now = std::time(nullptr);
        char date[64]; std::strftime(date, sizeof(date), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        std::fprintf(f_, "# pfsm-trajectory v1\n# date %s\n", date);
        std::fprintf(f_, "# config %s\n# config_fnv %016llx\n", m.config_path.c_str(),
                     (unsigned long long)fnv1a_file(m.config_path));
        std::fprintf(f_, "# mesh %s\n# mesh_fnv %016llx\n", m.mesh_path.c_str(),
                     (unsigned long long)fnv1a_file(m.mesh_path));
        std::fprintf(f_, "# tool_type %d\n# tool_mesh %s\n# tool_mesh_fnv %016llx\n", m.tool_type,
                     m.tool_mesh_path.empty() ? "-" : m.tool_mesh_path.c_str(),
                     (unsigned long long)(m.tool_mesh_path.empty() ? 0 : fnv1a_file(m.tool_mesh_path)));
        std::fprintf(f_, "# dt %.9g\n# substeps %d\n# num_iterations %d\n# knife_radius %.9g\n# N %d\n",
                     m.dt, m.substeps, m.num_iterations, m.knife_radius, m.N);
        std::fprintf(f_, "frame,t_wall,kx,ky,kz,r00,r01,r02,r10,r11,r12,r20,r21,r22,active,substeps\n");
        t0_ = std::chrono::steady_clock::now();
        std::printf("[record] writing trajectory to %s\n", path.c_str());
        return true;
    }
    void write(int frame, const float3& kp, const float rot[9], int active, int substeps) {
        if (!f_) return;
        double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
        // %.9g round-trips float32 exactly, so a replay feeds the solver bit-identical poses
        std::fprintf(f_, "%d,%.6f,%.9g,%.9g,%.9g", frame, t, kp.x, kp.y, kp.z);
        for (int k = 0; k < 9; k++) std::fprintf(f_, ",%.9g", rot[k]);
        std::fprintf(f_, ",%d,%d\n", active, substeps);
        ++n_; std::fflush(f_);   // every frame: survives Ctrl+C (no destructor), cost ~us
    }
    ~TrajectoryRecorder() {
        if (f_) { std::fclose(f_); std::printf("[record] closed after %d frames\n", n_); }
    }
    bool active() const { return f_ != nullptr; }
private:
    FILE* f_ = nullptr;
    int n_ = 0;
    std::chrono::steady_clock::time_point t0_;
};

// Per-source removal attribution + CPU stage times from simrec_frame::advance_frame.
// -1 in a count column = that stage does not exist on this path (tool_bench --gpu-only).
struct FrameSources { int gpu_new = -1, orphan_new = -1, bfs_new = -1; double orphan_ms = 0, bfs_ms = 0; };

class FrameStatsLogger {
public:
    bool open(const std::string& path, const std::vector<float3>& rest_pos, float avg_spacing,
              const std::string& provenance = "") {
        f_ = std::fopen(path.c_str(), "w");
        if (!f_) { std::fprintf(stderr, "[stats] cannot open %s\n", path.c_str()); return false; }
        spacing_ = avg_spacing;
        float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
        for (const auto& p : rest_pos) {
            mn[0] = std::fmin(mn[0], p.x); mn[1] = std::fmin(mn[1], p.y); mn[2] = std::fmin(mn[2], p.z);
            mx[0] = std::fmax(mx[0], p.x); mx[1] = std::fmax(mx[1], p.y); mx[2] = std::fmax(mx[2], p.z);
        }
        for (int k = 0; k < 3; k++) {
            float pad = 0.1f * (mx[k] - mn[k]);
            lo_[k] = mn[k] - pad; hi_[k] = mx[k] + pad;
        }
        prev_ = rest_pos;
        std::fprintf(f_, "# pfsm-frame-stats v2\n# avg_spacing %.9g\n# bbox_lo %.6g %.6g %.6g\n# bbox_hi %.6g %.6g %.6g\n",
                     spacing_, lo_[0], lo_[1], lo_[2], hi_[0], hi_[1], hi_[2]);
        if (!provenance.empty()) std::fprintf(f_, "%s", provenance.c_str());   // e.g. "# replay <file>\n"
        std::fprintf(f_, "frame,t_wall,frame_ms,gpu_ms,knife_active,knife_step,active,removed,new_removed,"
                         "contact,max_disp_sp,n_fast,n_out,obs_us,gpu_new,orphan_new,bfs_new,orphan_ms,bfs_ms\n");
        t0_ = last_ = std::chrono::steady_clock::now();
        std::printf("[stats] writing per-frame stats to %s\n", path.c_str());
        return true;
    }
    // Call at the very end of the frame (after orphan cleanup + BFS split).
    // contact = alive particles within knife_radius*(1+1e-4) of the sphere centre
    //           (sphere tool only; -1 for mesh-SDF tools) + particles removed this frame.
    void write(int frame, double gpu_ms, const float3& kp, float knife_radius, int knife_active,
               int tool_type, const std::vector<float3>& pos, const std::vector<int>& removed,
               const std::vector<int>& fixed, double extra_us = 0.0, const FrameSources& src = FrameSources()) {
        if (!f_) return;
        auto tb = std::chrono::steady_clock::now();
        double frame_ms = std::chrono::duration<double, std::milli>(tb - last_).count();
        last_ = tb;
        double t = std::chrono::duration<double>(tb - t0_).count();

        int N = (int)pos.size(), active = 0, rem = 0, contact = 0, n_fast = 0, n_out = 0;
        float max_disp = 0.0f;
        float r2 = knife_radius * (1.0f + 1e-4f); r2 *= r2;
        for (int i = 0; i < N; i++) {
            if (removed[i]) { rem++; continue; }
            active++;
            const float3& p = pos[i];
            if (knife_active && tool_type == 0) {
                float dx = p.x - kp.x, dy = p.y - kp.y, dz = p.z - kp.z;
                if (dx*dx + dy*dy + dz*dz < r2) contact++;
            }
            if (!fixed[i]) {
                float dx = p.x - prev_[i].x, dy = p.y - prev_[i].y, dz = p.z - prev_[i].z;
                float d = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (d > max_disp) max_disp = d;
                if (d > spacing_) n_fast++;
            }
            if (p.x < lo_[0] || p.x > hi_[0] || p.y < lo_[1] || p.y > hi_[1] || p.z < lo_[2] || p.z > hi_[2]) n_out++;
        }
        int new_rem = (rem_prev_ < 0) ? 0 : rem - rem_prev_;
        rem_prev_ = rem;
        if (tool_type == 0) contact += (knife_active ? new_rem : 0); else contact = -1;
        float step = have_kp_ ? std::sqrt((kp.x-kp_prev_.x)*(kp.x-kp_prev_.x) + (kp.y-kp_prev_.y)*(kp.y-kp_prev_.y) +
                                          (kp.z-kp_prev_.z)*(kp.z-kp_prev_.z)) : 0.0f;
        kp_prev_ = kp; have_kp_ = true;
        prev_ = pos;

        // obs_us = this stats pass + the recorder write of the same frame (extra_us)
        double obs_us = extra_us + std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - tb).count();
        std::fprintf(f_, "%d,%.6f,%.3f,%.3f,%d,%.6g,%d,%d,%d,%d,%.4f,%d,%d,%.1f,%d,%d,%d,%.3f,%.3f\n",
                     frame, t, frame_ms, gpu_ms, knife_active, step, active, rem, new_rem, contact,
                     spacing_ > 0 ? max_disp / spacing_ : 0.0f, n_fast, n_out, obs_us,
                     src.gpu_new, src.orphan_new, src.bfs_new, src.orphan_ms, src.bfs_ms);
        ++n_; std::fflush(f_);   // every frame: survives Ctrl+C (no destructor), cost ~us
    }
    ~FrameStatsLogger() {
        if (f_) { std::fclose(f_); std::printf("[stats] closed after %d frames\n", n_); }
    }
    bool active() const { return f_ != nullptr; }
private:
    FILE* f_ = nullptr;
    int n_ = 0, rem_prev_ = -1;
    float spacing_ = 0, lo_[3], hi_[3];
    std::vector<float3> prev_;
    float3 kp_prev_; bool have_kp_ = false;
    std::chrono::steady_clock::time_point t0_, last_;
};

// ---------------------------------------------------------------------------------------------
// Step 0.5b — TrajectoryReader (--replay <file>): parses a TrajectoryRecorder file and checks
// its header against the current run. Mismatches are errors unless allow_mismatch is set
// (needed later to replay the SAME hand trajectory with a DIFFERENT tool/config on purpose);
// mismatches are then returned so they can be logged in the stats provenance header.
struct ReplayFrame { float3 kp; float rot[9]; int active; int substeps; };

class TrajectoryReader {
public:
    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f) { std::fprintf(stderr, "[replay] cannot open %s\n", path.c_str()); return false; }
        path_ = path;
        std::string line;
        bool have_cols = false;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                size_t sp = line.find(' ', 2);
                if (line.size() > 2 && sp != std::string::npos)
                    meta_.emplace_back(line.substr(2, sp - 2), line.substr(sp + 1));
                continue;
            }
            if (!have_cols) { have_cols = true; continue; }       // column header row
            ReplayFrame r{};
            int frame = 0; double t = 0;
            int n = std::sscanf(line.c_str(), "%d,%lf,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d",
                                &frame, &t, &r.kp.x, &r.kp.y, &r.kp.z,
                                &r.rot[0], &r.rot[1], &r.rot[2], &r.rot[3], &r.rot[4], &r.rot[5],
                                &r.rot[6], &r.rot[7], &r.rot[8], &r.active, &r.substeps);
            if (n != 16) { std::fprintf(stderr, "[replay] bad line: %s\n", line.c_str()); return false; }
            if (frame != (int)frames_.size()) {
                std::fprintf(stderr, "[replay] non-contiguous frame %d (expected %zu)\n", frame, frames_.size());
                return false;
            }
            frames_.push_back(r);
        }
        if (frames_.empty()) { std::fprintf(stderr, "[replay] no frames in %s\n", path.c_str()); return false; }
        std::printf("[replay] loaded %zu frames from %s\n", frames_.size(), path.c_str());
        return true;
    }
    std::string get(const std::string& key) const {
        for (auto& kv : meta_) if (kv.first == key) return kv.second;
        return "";
    }
    // Compare against the current run. Returns the list of mismatch descriptions (empty = OK).
    std::vector<std::string> check(const RecordMeta& m) const {
        std::vector<std::string> bad;
        auto hex = [](uint64_t h) { char b[32]; std::snprintf(b, sizeof(b), "%016llx", (unsigned long long)h); return std::string(b); };
        auto cmp = [&](const std::string& key, const std::string& now) {
            std::string rec = get(key);
            if (rec != now) bad.push_back(key + ": recorded=" + rec + " current=" + now);
        };
        auto num = [](double v) { char b[64]; std::snprintf(b, sizeof(b), "%.9g", v); return std::string(b); };
        cmp("config_fnv", hex(fnv1a_file(m.config_path)));
        cmp("mesh_fnv", hex(fnv1a_file(m.mesh_path)));
        cmp("tool_type", std::to_string(m.tool_type));
        cmp("tool_mesh_fnv", hex(m.tool_mesh_path.empty() ? 0 : fnv1a_file(m.tool_mesh_path)));
        cmp("dt", num(m.dt));
        cmp("substeps", std::to_string(m.substeps));
        cmp("num_iterations", std::to_string(m.num_iterations));
        cmp("knife_radius", num(m.knife_radius));
        cmp("N", std::to_string(m.N));
        return bad;
    }
    const std::vector<ReplayFrame>& frames() const { return frames_; }
    const std::string& path() const { return path_; }
private:
    std::string path_;
    std::vector<std::pair<std::string, std::string>> meta_;
    std::vector<ReplayFrame> frames_;
};

}  // namespace simrec
