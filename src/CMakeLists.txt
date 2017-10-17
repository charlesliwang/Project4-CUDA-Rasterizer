set(SOURCE_FILES
    "rasterize.cu"
    "rasterize.h"
    "rasterizeTools.h"
    "timer.h"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )

LIST(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61")