#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

// Define the communication and execution spaces
using Co = KokkosComm::DefaultCommunicationSpace;
using Ex = Kokkos::DefaultExecutionSpace;

// Source rank
int src_rank = 1;

// Create a communicator
auto comm = KokkosComm::Communicator<Co, Ex>::duplicate(raw_comm_handle, exec_space);

// Allocate a view to receive the data
Kokkos::View<double*> data("recv_view", 100);

// Initiate a non-blocking receive with a handle
auto req1 = KokkosComm::recv(comm, data, src_rank);

// Simulate a blocking receive by waiting immediately
KokkosComm::recv(comm, data, src_rank).wait();

// Wait for a requests to complete
KokkosComm::wait(req1);
