cmake_minimum_required(VERSION 3.20)
project(LearnCuda CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

# blow code is very import for debug cuda device code
# set(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -G")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

set(CMAKE_CUDA_FLAGS "-Wall")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75")


file(GLOB files *.cu)
foreach (file ${files})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" exe ${file})
    add_executable(${exe} ${file})
    set_target_properties(${exe} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${exe} PROPERTIES CUDA_ARCHITECTURES "75")
    target_link_libraries(${exe} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
    message(--\ src/${exe}.cu\ will\ be\ compiled\ to\ bin/${exe})
endforeach ()
