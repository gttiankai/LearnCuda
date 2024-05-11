cmake_minimum_required(VERSION 3.20)
project(LearnCuda CUDA CXX)
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()
find_package(CUDA REQUIRED)


set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_FLAGS "-Wall")
# blow code is very import for debug cuda device code
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

# config cuda architectures SM
set(SM_SETS 70 75 80 86 89 90)
set(FIND_SM False)
foreach (SM_NUM IN LISTS SM_SETS)
    string(FIND "${SM}" "${SM_NUM}" SM_POS)
    if (SM_POS GREATER -1)
        if (FIND_SM STREQUAL False)
            set(ENV{TORCH_CUDA_ARCH_LIST} "")
        endif ()
        set(FIND_SM True)
        # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM_NUM},code=\\\"sm_${SM_NUM},compute_${SM_NUM}\\\"")
        if (SM_NUM STREQUAL 70 OR SM_NUM STREQUAL 75 OR SM_NUM STREQUAL 80 OR SM_NUM STREQUAL 86 OR SM_NUM STREQUAL 89 OR SM_NUM STREQUAL 90)
            set(USING_WMMA True)
        endif ()
        list(APPEND SUPPORT_ARCHITECTURES ${SM_NUM})
    endif ()
endforeach ()
message("-- support architectures: ${SUPPORT_ARCHITECTURES}")


file(GLOB files *.cu)
foreach (file ${files})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" exe ${file})
    add_executable(${exe} ${file})
    set_target_properties(${exe} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${exe} PROPERTIES CUDA_ARCHITECTURES "${SUPPORT_ARCHITECTURES}")
    target_link_libraries(${exe} PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
    target_compile_options(${exe} PRIVATE "--ptxas-options=-v,-warn-lmem-usage,--warn-on-spills")
    message(--\ src/${exe}.cu\ will\ be\ compiled\ to\ bin/${exe})
endforeach ()
