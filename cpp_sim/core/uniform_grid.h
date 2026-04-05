#pragma once

#include "particles.h"
#include <unordered_map>
#include <vector>
#include <cmath>

class UniformGrid {
public:
    void build(const std::vector<Vec3>& positions, double cell_size) {
        m_cell_size = cell_size;
        m_cells.clear();
        for (int i = 0; i < static_cast<int>(positions.size()); i++) {
            int64_t h = cell_hash(positions[i]);
            m_cells[h].push_back(i);
        }
    }

    void build(const Particles& particles, double cell_size) {
        m_cell_size = cell_size;
        m_cells.clear();
        for (int i = 0; i < particles.size(); i++) {
            int64_t h = cell_hash(particles[i].pos);
            m_cells[h].push_back(i);
        }
    }

    // 查询球形范围内的所有粒子 id
    std::vector<int> query_sphere(const Vec3& center, double radius) const {
        std::vector<int> result;
        double r2 = radius * radius;

        int cx = to_cell(center.x());
        int cy = to_cell(center.y());
        int cz = to_cell(center.z());
        int r_cells = static_cast<int>(std::ceil(radius / m_cell_size)) + 1;

        for (int dx = -r_cells; dx <= r_cells; dx++) {
            for (int dy = -r_cells; dy <= r_cells; dy++) {
                for (int dz = -r_cells; dz <= r_cells; dz++) {
                    int64_t h = hash(cx + dx, cy + dy, cz + dz);
                    auto it = m_cells.find(h);
                    if (it == m_cells.end()) continue;
                    for (int pid : it->second) {
                        // 实际距离检查在调用方做（或这里加上位置数组也行）
                        result.push_back(pid);
                    }
                }
            }
        }
        return result;
    }

    // 带精确距离过滤的查询
    std::vector<int> query_sphere(const Vec3& center, double radius,
                                  const std::vector<Vec3>& positions) const {
        std::vector<int> result;
        double r2 = radius * radius;

        int cx = to_cell(center.x());
        int cy = to_cell(center.y());
        int cz = to_cell(center.z());
        int r_cells = static_cast<int>(std::ceil(radius / m_cell_size)) + 1;

        for (int dx = -r_cells; dx <= r_cells; dx++) {
            for (int dy = -r_cells; dy <= r_cells; dy++) {
                for (int dz = -r_cells; dz <= r_cells; dz++) {
                    int64_t h = hash(cx + dx, cy + dy, cz + dz);
                    auto it = m_cells.find(h);
                    if (it == m_cells.end()) continue;
                    for (int pid : it->second) {
                        if ((positions[pid] - center).squaredNorm() <= r2) {
                            result.push_back(pid);
                        }
                    }
                }
            }
        }
        return result;
    }

    // 同上，但用 Particles
    std::vector<int> query_sphere(const Vec3& center, double radius,
                                  const Particles& particles) const {
        std::vector<int> result;
        double r2 = radius * radius;

        int cx = to_cell(center.x());
        int cy = to_cell(center.y());
        int cz = to_cell(center.z());
        int r_cells = static_cast<int>(std::ceil(radius / m_cell_size)) + 1;

        for (int dx = -r_cells; dx <= r_cells; dx++) {
            for (int dy = -r_cells; dy <= r_cells; dy++) {
                for (int dz = -r_cells; dz <= r_cells; dz++) {
                    int64_t h = hash(cx + dx, cy + dy, cz + dz);
                    auto it = m_cells.find(h);
                    if (it == m_cells.end()) continue;
                    for (int pid : it->second) {
                        if ((particles[pid].pos - center).squaredNorm() <= r2) {
                            result.push_back(pid);
                        }
                    }
                }
            }
        }
        return result;
    }

private:
    double m_cell_size = 0.1;
    std::unordered_map<int64_t, std::vector<int>> m_cells;

    int to_cell(double v) const {
        return static_cast<int>(std::floor(v / m_cell_size));
    }

    int64_t cell_hash(const Vec3& pos) const {
        return hash(to_cell(pos.x()), to_cell(pos.y()), to_cell(pos.z()));
    }

    static int64_t hash(int x, int y, int z) {
        // XOR 素数哈希
        return static_cast<int64_t>(x) * 73856093LL
             ^ static_cast<int64_t>(y) * 19349663LL
             ^ static_cast<int64_t>(z) * 83492791LL;
    }
};
