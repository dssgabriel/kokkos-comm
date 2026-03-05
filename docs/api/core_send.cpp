#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

// Define the communication and execution spaces
using Co = KokkosComm::DefaultCommunicationSpace;
using Ex = Kokkos::DefaultExecutionSpace;

// Create an execution space instance
auto exec = Ex();
// Create a communicator
auto comm = KokkosComm::Communicator<Co, Ex>::duplicate(raw_comm_handle, exec);

// Create a Kokkos view
Kokkos::View<double*> data("data", 100);

// Fill the view with some data
Kokkos::parallel_for("fill_data", Kokkos::RangePolicy(exec, 0, 100), KOKKOS_LAMBDA(int i) {
  data(i) = static_cast<double>(i);
});
exec.fence();

// Destination rank
int dst_rank = 1;

// Initiate a non-blocking send with a handle
auto req1 = KokkosComm::send(comm, data, dst_rank);

// Simulate a blocking send by waiting immediately
KokkosComm::send(comm, data, dst_rank).wait();

// Wait for a requests to complete
KokkosComm::wait(req1);
