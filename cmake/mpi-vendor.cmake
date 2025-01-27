function(kokkoscomm_set_mpi_vendor_variables)
    # Initialize the variables to false
    set(KOKKOSCOMM_IMPL_MPI_IS_MPICH FALSE CACHE BOOL "Is MPI MPICH")
    set(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI FALSE CACHE BOOL "Is MPI OpenMPI")

    if(KOKKOSCOMM_ENABLE_MPI)
        if(NOT MPIEXEC_EXECUTABLE)
            message(WARNING "MPIEXEC_EXECUTABLE is not set - unable to determine MPI vendor")
            return()
        endif()

        # Get the directory of the MPI executable
        get_filename_component(MPIEXEC_DIR ${MPIEXEC_EXECUTABLE} DIRECTORY)

        # Check for mpichversion
        find_program(MPICHVERSION_EXECUTABLE mpichversion HINTS ${MPIEXEC_DIR})
        find_program(OMPI_INFO_EXECUTABLE ompi_info HINTS ${MPIEXEC_DIR})

        if (MPICHVERSION_EXECUTABLE AND OPENMPI_INFO_EXECUTABLE)
            message(WARNING "Both MPICHVERSION_EXECUTABLE and OMPI_INFO_EXECUTABLE are set - unable to determine MPI vendor")
        elseif(MPICHVERSION_EXECUTABLE)
            message(STATUS "Detected MPI as MPICH")
            set(KOKKOSCOMM_IMPL_MPI_IS_MPICH TRUE CACHE BOOL "Is MPI MPICH" FORCE)
        elseif(OMPI_INFO_EXECUTABLE)
            message(STATUS "Detected MPI as OpenMPI")
            set(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI TRUE CACHE BOOL "Is MPI OpenMPI" FORCE)
        endif()
    endif()
endfunction()
