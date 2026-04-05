#pragma once

#include <Eigen/Dense>
#include <vector>

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

struct Particle {
    Vec3 pos;
    Vec3 vel;
    Vec3 predicted_pos;
    double mass = 1.0;
    double inv_mass = 1.0;
    int id = 0;
    int layer_id = -1;  // 用于评估：0=肿瘤核心, 1=粘连界面, 2=神经鞘, -1=未标注
    bool removed = false;  // 被 CUSA 吸走的碎片
    bool fixed = false;    // 固定粒子（不受力/不移动）
};

struct Particles {
    std::vector<Particle> data;

    int size() const { return static_cast<int>(data.size()); }

    Particle& operator[](int i) { return data[i]; }
    const Particle& operator[](int i) const { return data[i]; }

    int add(const Vec3& pos, double mass = 1.0, int layer_id = -1) {
        int id = size();
        Particle p;
        p.pos = pos;
        p.vel = Vec3::Zero();
        p.predicted_pos = pos;
        p.mass = mass;
        p.inv_mass = (mass > 0) ? 1.0 / mass : 0.0;
        p.id = id;
        p.layer_id = layer_id;
        data.push_back(p);
        return id;
    }
};
