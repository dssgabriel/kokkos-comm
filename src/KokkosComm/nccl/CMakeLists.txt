target_sources(
  KokkosComm
  INTERFACE
    # Public NCCL headers
    FILE_SET HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES
        # Objects
        nccl_space.hpp
        communicator.hpp
        request.hpp
        # Point-to-point
        send.hpp
        recv.hpp
        # Collectives
        broadcast.hpp
        allgather.hpp
        alltoall.hpp
        allreduce.hpp
        reduce.hpp
    # Implementation detail NCCL headers
    FILE_SET HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES impl/error_handling.hpp
)

target_link_libraries(KokkosComm INTERFACE NCCL::NCCL)
