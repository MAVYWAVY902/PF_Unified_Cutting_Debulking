#include "core/particles.h"
#include "core/cluster.h"
#include "core/tet_mesh.h"
#include "solver/shape_matching_cpu.h"
#include <iostream>
#include <cmath>

int main() {
    // 测试：tet mesh 落地后的角动量
    std::cout << "===== Tet mesh 落地测试 =====" << std::endl;

    TetMesh tm;
    tm.load_msh("/home/yunxin/xpbd-tissue-sim/resource/cube/cube4.msh");

    ShapeMatchingSolver s;

    // 抬高 mesh
    double min_y = 1e10;
    for (auto& v : tm.vertices) min_y = std::min(min_y, v.y());
    double y_offset = -min_y + 0.5;

    for (int i = 0; i < tm.num_vertices(); i++) {
        Vec3 pos = tm.vertices[i] + Vec3(0, y_offset, 0);
        s.particles.add(pos, 1.0, 0);
    }

    s.clusters = tm.build_clusters_from_tets(s.particles, 2);
    s.config.dt = 0.005;
    s.config.num_iterations = 5;
    s.config.stiffness = 0.8;
    s.config.enable_floor = true;
    s.config.floor_y = 0.0;

    int N = s.particles.size();

    for (int step = 0; step < 400; step++) {
        s.step();

        if (step % 20 == 0) {
            Vec3 cm = Vec3::Zero();
            for (int i = 0; i < N; i++) cm += s.particles[i].pos;
            cm /= N;

            Vec3 ang = Vec3::Zero();
            double ke = 0;
            for (int i = 0; i < N; i++) {
                Vec3 r = s.particles[i].pos - cm;
                ang += r.cross(s.particles[i].vel);
                ke += 0.5 * s.particles[i].vel.squaredNorm();
            }

            printf("step %3d | cm=(%.3f,%.3f,%.3f) | ang=(%+.4f,%+.4f,%+.4f) | KE=%.4f\n",
                step, cm.x(), cm.y(), cm.z(), ang.x(), ang.y(), ang.z(), ke);
        }
    }

    // 同样测试粒子立方体
    std::cout << "\n===== 粒子立方体落地测试 =====" << std::endl;
    {
        ShapeMatchingSolver s2;
        double spacing = 0.25;  // 匹配 cube4 的尺寸
        int side = 5;
        for (int x = 0; x < side; x++)
            for (int y = 0; y < side; y++)
                for (int z = 0; z < side; z++)
                    s2.particles.add(Vec3(x * spacing, y * spacing + 0.5, z * spacing), 1.0, 0);

        s2.clusters = build_overlapping_clusters(s2.particles, spacing * 2.5);
        s2.config.dt = 0.005;
        s2.config.num_iterations = 5;
        s2.config.stiffness = 0.8;
        s2.config.enable_floor = true;
        s2.config.floor_y = 0.0;

        int N2 = s2.particles.size();
        for (int step = 0; step < 400; step++) {
            s2.step();
            if (step % 20 == 0) {
                Vec3 cm = Vec3::Zero();
                for (int i = 0; i < N2; i++) cm += s2.particles[i].pos;
                cm /= N2;

                Vec3 ang = Vec3::Zero();
                double ke = 0;
                for (int i = 0; i < N2; i++) {
                    Vec3 r = s2.particles[i].pos - cm;
                    ang += r.cross(s2.particles[i].vel);
                    ke += 0.5 * s2.particles[i].vel.squaredNorm();
                }
                printf("step %3d | cm=(%.3f,%.3f,%.3f) | ang=(%+.4f,%+.4f,%+.4f) | KE=%.4f\n",
                    step, cm.x(), cm.y(), cm.z(), ang.x(), ang.y(), ang.z(), ke);
            }
        }
    }

    return 0;
}
