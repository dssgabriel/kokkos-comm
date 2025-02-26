*************
Building MPIs
*************

You may find that you need an MPI to test Kokkos Comm against.
Here are some informal notes about configurations that have worked in the past.

MPICH 4.2.0
===========

With CUDA 12.2, it's been observed that the default "device" ``ch4:ofi`` results in intermittent ``MPI_Recv`` errors.
Setting ``--with-device=ch4:ucx`` sseems to avoid this.
Unclear at the time of writing whether this is a subtle bug in Kokkos Comm, or a problem with MPICH, or a problem with the environment MPICH was installed in.

.. code-block:: bash

    MPICH_SRC="$PWD"/mpich-4.2.0
    MPI_INSTALL="$PWD"/mpich-4.2.0-install
    wget --continue https://www.mpich.org/static/downloads/4.2.0/mpich-4.2.0.tar.gz
    tar -xf mpich-4.2.0.tar.gz
    rm -rf "$MPICH_INSTALL"
    (cd $MPICH_SRC; ./configure --prefix="$MPICH_INSTALL" --with-cuda=/usr/local/cuda/ --enable-error-messages=all --with-device=ch4:ucx)
    (cd $MPICH_SRC; make -j $(nproc) install)


Open MPI 5.0.6
==============

This was observed to work with CUDA 12.2.
It was necessary to use both ``--with-cuda`` and ``--with-cuda-libdir``, otherwise it would not detect the CUDA library.

.. code-block:: bash
    
    OPENMPI_SRC="$PWD"/openmpi-5.0.6
    MPI_INSTALL="$PWD"/openmpi-5.0.6-install
    wget --continue https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.6.tar.gz
    tar -xf openmpi-5.0.6.tar.gz
    rm -rf "$OPENMPI_INSTALL"
    (cd $OPENMPI_SRC; ./configure --prefix="$OPENMPI_INSTALL" --enable-mpi-ext=cuda --with-cuda=/usr/local/cuda --with-cuda-libdir=/usr/local/cuda/lib64/stubs)
    (cd $OPENMPI_SRC; make -j $(nproc))
    (cd $OPENMPI_SRC; make install)