#include "core/particles.h"
#include "core/cluster.h"
#include "core/tet_mesh.h"
#include "solver/shape_matching_cpu.h"
#include <iostream>
#include <cmath>

int main() {
    ShapeMatchingSolver solver;
    TetMesh tet_mesh;

    // 测试 1：简单粒子立方体（已知正确的基准）
    std::cout << "===== 测试 1：粒子立方体 =====" << std::endl;
    {
        ShapeMatchingSolver s;
        double spacing = 0.02;
        int side = 5;
        for (int x = 0; x < side; x++)
            for (int y = 0; y < side; y++)
                for (int z = 0; z < side; z++)
                    s.particles.add(Vec3(x, y + 10, z) * spacing, 1.0, 0);  // 高空

        s.clusters = build_overlapping_clusters(s.particles, spacing * 2.5);
        s.config.dt = 0.005;
        s.config.num_iterations = 5;
        s.config.stiffness = 0.8;
        s.config.enable_floor = false;  // 先不开地板，纯自由落体

        Vec3 cm0 = Vec3::Zero();
        for (int i = 0; i < s.particles.size(); i++) cm0 += s.particles[i].pos;
        cm0 /= s.particles.size();

        for (int step = 0; step < 20; step++) {
            s.step();
            Vec3 cm = Vec3::Zero();
            Vec3 ang = Vec3::Zero();  // 角动量
            for (int i = 0; i < s.particles.size(); i++) {
                cm += s.particles[i].pos;
            }
            cm /= s.particles.size();

            for (int i = 0; i < s.particles.size(); i++) {
                Vec3 r = s.particles[i].pos - cm;
                ang += r.cross(s.particles[i].vel);
            }

            if (step % 5 == 0) {
                printf("  step %2d | cm=(%.4f, %.4f, %.4f) | ang_mom=(%.6f, %.6f, %.6f)\n",
                    step, cm.x(), cm.y(), cm.z(), ang.x(), ang.y(), ang.z());
            }
        }
        std::cout << "  → 角动量应该接近 0（纯平移，无旋转）" << std::endl;
    }

    // 测试 2：tet mesh 自由落体（无地板）
    std::cout << "\n===== 测试 2：Tet mesh 自由落体 =====" << std::endl;
    {
        TetMesh tm;
        if (!tm.load_msh("/home/yunxin/xpbd-tissue-sim/resource/cube/cube4.msh")) {
            std::cerr << "加载失败" << std::endl;
            return 1;
        }

        ShapeMatchingSolver s;
        for (int i = 0; i < tm.num_vertices(); i++) {
            s.particles.add(tm.vertices[i], 1.0, 0);
        }
        s.clusters = tm.build_clusters_from_tets(s.particles, 2);
        s.config.dt = 0.005;
        s.config.num_iterations = 5;
        s.config.stiffness = 0.8;
        s.config.enable_floor = false;

        for (int step = 0; step < 20; step++) {
            s.step();
            Vec3 cm = Vec3::Zero();
            Vec3 ang = Vec3::Zero();
            for (int i = 0; i < s.particles.size(); i++) cm += s.particles[i].pos;
            cm /= s.particles.size();
            for (int i = 0; i < s.particles.size(); i++) {
                Vec3 r = s.particles[i].pos - cm;
                ang += r.cross(s.particles[i].vel);
            }
            if (step % 5 == 0) {
                printf("  step %2d | cm=(%.4f, %.4f, %.4f) | ang_mom=(%.6f, %.6f, %.6f)\n",
                    step, cm.x(), cm.y(), cm.z(), ang.x(), ang.y(), ang.z());
            }
        }
        std::cout << "  → 角动量应该接近 0" << std::endl;
    }

    // 测试 3：检查簇的对称性
    std::cout << "\n===== 测试 3：簇对称性检查 =====" << std::endl;
    {
        TetMesh tm;
        tm.load_msh("/home/yunxin/xpbd-tissue-sim/resource/cube/cube4.msh");

        ShapeMatchingSolver s;
        for (int i = 0; i < tm.num_vertices(); i++) {
            s.particles.add(tm.vertices[i], 1.0, 0);
        }
        s.clusters = tm.build_clusters_from_tets(s.particles, 2);

        // 检查前 5 个簇的 rest_cm 和 rest_offsets
        for (int c = 0; c < std::min(5, (int)s.clusters.size()); c++) {
            auto& cl = s.clusters[c];
            Vec3 check_cm = Vec3::Zero();
            for (int j = 0; j < (int)cl.particle_ids.size(); j++) {
                check_cm += cl.rest_offsets[j];
            }
            printf("  簇 %d: %d 粒子 | rest_cm=(%.4f,%.4f,%.4f) | offset_sum=(%.6f,%.6f,%.6f)\n",
                c, (int)cl.particle_ids.size(),
                cl.rest_cm.x(), cl.rest_cm.y(), cl.rest_cm.z(),
                check_cm.x(), check_cm.y(), check_cm.z());
        }
        std::cout << "  → offset_sum 应该接近 0（质心减掉了）" << std::endl;
    }

    // 测试 4：单个簇 solve 测试（纯平移）
    std::cout << "\n===== 测试 4：单个簇 solve 测试 =====" << std::endl;
    {
        // 创建一个简单的 4 粒子四面体
        Particles p;
        p.add(Vec3(0, 0, 0));
        p.add(Vec3(1, 0, 0));
        p.add(Vec3(0.5, 0.866, 0));
        p.add(Vec3(0.5, 0.289, 0.816));

        Cluster c;
        c.particle_ids = {0, 1, 2, 3};
        c.init(p);

        // 纯平移：所有粒子移动 (1, 2, 3)
        std::vector<Vec3> positions(4);
        Vec3 translate(1, 2, 3);
        for (int i = 0; i < 4; i++) {
            positions[i] = p[i].pos + translate;
        }

        std::vector<Vec3> deltas(4, Vec3::Zero());
        std::vector<int> counts(4, 0);
        c.solve(positions, deltas, counts);

        std::cout << "  纯平移后 deltas (应该全是 0):" << std::endl;
        for (int i = 0; i < 4; i++) {
            printf("    粒子 %d: delta=(%.6f, %.6f, %.6f) count=%d\n",
                i, deltas[i].x(), deltas[i].y(), deltas[i].z(), counts[i]);
        }

        // 测试旋转：90度绕 Y 轴
        std::cout << "  90度 Y 旋转后 deltas:" << std::endl;
        Mat3 rot;
        rot << 0, 0, 1,
               0, 1, 0,
              -1, 0, 0;
        Vec3 rest_cm = Vec3::Zero();
        for (int i = 0; i < 4; i++) rest_cm += p[i].pos;
        rest_cm /= 4;

        for (int i = 0; i < 4; i++) {
            positions[i] = rot * (p[i].pos - rest_cm) + rest_cm + translate;
        }

        std::fill(deltas.begin(), deltas.end(), Vec3::Zero());
        std::fill(counts.begin(), counts.end(), 0);
        c.solve(positions, deltas, counts);

        for (int i = 0; i < 4; i++) {
            printf("    粒子 %d: delta=(%.6f, %.6f, %.6f)\n",
                i, deltas[i].x(), deltas[i].y(), deltas[i].z());
        }
        std::cout << "  → deltas 应该把粒子拉回原始形状" << std::endl;
    }

    return 0;
}
