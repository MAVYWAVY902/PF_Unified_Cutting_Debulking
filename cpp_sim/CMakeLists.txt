cmake_minimum_required(VERSION 3.18)
project(pf_shape_matching LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# OpenMP
find_package(OpenMP)

# OpenGL + GLFW + GLEW
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 3.3 REQUIRED)

# CUDA
find_package(CUDAToolkit REQUIRED)

# ===== CPU 版本 =====
add_executable(pf_sim main.cpp)
target_include_directories(pf_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pf_sim PRIVATE
    Eigen3::Eigen
    OpenGL::GL
    GLEW::GLEW
    glfw
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pf_sim PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_options(pf_sim PRIVATE -O3 -march=native)

# ===== GPU 版本 =====
add_executable(pf_sim_gpu
    main_gpu.cpp
    solver/shape_matching_gpu.cu
)
target_include_directories(pf_sim_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/svd3/svd3_cuda
)
target_link_libraries(pf_sim_gpu PRIVATE
    Eigen3::Eigen
    OpenGL::GL
    GLEW::GLEW
    glfw
    CUDA::cudart
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pf_sim_gpu PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(pf_sim_gpu PROPERTIES
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES "70;75;80;86"
)
target_compile_options(pf_sim_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# ===== GPU + Viewer 版本 =====
add_executable(pf_sim_gpu_viewer
    main_gpu_viewer.cpp
    solver/shape_matching_gpu.cu
)
target_include_directories(pf_sim_gpu_viewer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/svd3/svd3_cuda
)
find_package(yaml-cpp REQUIRED)
target_link_libraries(pf_sim_gpu_viewer PRIVATE
    Eigen3::Eigen
    OpenGL::GL
    GLEW::GLEW
    glfw
    CUDA::cudart
    yaml-cpp
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pf_sim_gpu_viewer PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(pf_sim_gpu_viewer PROPERTIES
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES "70;75;80;86"
)
target_compile_options(pf_sim_gpu_viewer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# ===== GPU 验证 =====
add_executable(verify_gpu
    verify_gpu.cpp
    solver/shape_matching_gpu.cu
)
target_include_directories(verify_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/svd3/svd3_cuda
)
target_link_libraries(verify_gpu PRIVATE
    Eigen3::Eigen
    CUDA::cudart
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(verify_gpu PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(verify_gpu PROPERTIES
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES "75"
)
target_compile_options(verify_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# ===== Debug tumor =====
add_executable(debug_tumor
    debug_tumor.cpp
    solver/shape_matching_gpu.cu
)
target_include_directories(debug_tumor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/svd3/svd3_cuda
)
target_link_libraries(debug_tumor PRIVATE
    Eigen3::Eigen
    CUDA::cudart
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(debug_tumor PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(debug_tumor PROPERTIES
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES "75"
)
target_compile_options(debug_tumor PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)
