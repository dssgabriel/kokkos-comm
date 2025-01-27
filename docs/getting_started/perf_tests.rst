**********
Benchmarks
**********

Benchmarks are implemented in the ``perf_tests`` directory. Currently, all benchmarks use the MPI backend.

Available benchmarks:

.. list-table::
    :header-rows: 1
    :align: left
    :widths: 40 80

    * - File name
      - Description

    * - ``test_2dhalo.cpp``
      - a 2D halo exchange

    * - ``test_sendrecv.cpp``
      - ping-pong between ranks 0 and 1


To build and run the benchmarks, see the `dedicated page on testing <../dev/testing.html>`_: 
