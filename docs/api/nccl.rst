************************
Low-level NCCL interfaces
************************

.. list-table:: NCCL API Support
    :widths: 40 50 30
    :header-rows: 1

    * - NCCL routines
      - ``KokkosComm::Experimental::nccl::`` namespace
      - ``Kokkos::View`` support
    * - ``ncclSend``
      - ``send``
      - ✓
    * - ``ncclRecv``
      - ``recv``
      - ✓


Point-to-point
==============

The NCCL backend only works with ``Kokkos::Cuda`` as the execution space. Operations are enqueued on the CUDA stream and do not require explicit synchronization prior to submission.

.. cpp:namespace:: KokkosComm::Experimental::nccl

.. cpp:function:: template <KokkosView SendV> \
                  auto send(Communicator<NcclSpace, Kokkos::Cuda> &comm, const SendV &sv, int dst) -> Request<NcclSpace>

    Initiates a non-blocking send operation.

    :tparam SendV: A Kokkos View type.

    :param comm: The communicator handle associated with the operation.
    :param sv: The View to send.
    :param dst: The destination rank.

    :return: A Request object representing the non-blocking send operation.


.. cpp:function:: template <KokkosView RecvV> \
                  auto recv(Communicator<NcclSpace, Kokkos::Cuda> &comm, const RecvV &rv, int src) -> Request<NcclSpace>

    Initiates a non-blocking receive operation.

    :tparam RecvV: A Kokkos View type.

    :param comm: The communicator handle associated with the operation.
    :param rv: The View to receive into.
    :param src: The source rank.

    :return: A Request object representing the non-blocking receive operation.
