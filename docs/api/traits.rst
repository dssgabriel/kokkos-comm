*******************
Concepts and Traits
*******************

Concepts
========

.. cpp:namespace:: KokkosComm

.. cpp:concept:: template <typename T> KokkosView

    Specifies that a type ``T`` is a ``Kokkos::View`` object.


.. cpp:concept:: template <typename T> KokkosExecutionSpace

    Specifies that a type ``T`` is a ``Kokkos::ExecutionSpace``.


.. cpp:concept:: template <typename T> CommunicationSpace

    Specifies that a type ``T`` is a KokkosComm communication backend.


Traits
======

General traits
--------------

.. cpp:namespace:: KokkosComm

.. cpp:struct:: template<KokkosView View> Traits<View>

    A struct that can be specialized to implement custom behavior for a particular Kokkos view.

    .. cpp:type:: non_const_packed_view_type = Kokkos::View<typename View::non_const_data_type, typename View::array_layout, typename View::memory_space>

    .. cpp:type:: packed_view_type = Kokkos::View<typename View::data_type, typename View::array_layout, typename View::memory_space>


.. cpp:function:: template <KokkosView View> \
                  static auto data_handle(const View &v) -> View::pointer_type

    :tparam View: The type of the Kokkos view.

    :param v: The Kokkos view to query.

    :returns: The pointer to the underlying data allocation.


.. cpp:function:: template <KokkosView View> \
                           static auto span(const View &v) -> size_t

    :tparam View: The type of the Kokkos view.

    :param v: The Kokkos view to query.

    :returns: The number of bytes between the beginning of the first byte and the end of the last byte of data in ``v``.

    For example, if ``View`` was an std::vector<int16_t> of size 3, it would be 6.
    If the ``View`` is non-contiguous, the result includes any "holes" in ``v``.


.. cpp:function:: template <KokkosView View> \
                  static auto is_contiguous(const View &v) -> bool

    Checks if a view is contiguous.

    :tparam View: The type of the Kokkos view.

    :param v: The Kokkos view to query.

    :returns: True if, and only if, the data in ``v`` is contiguous.


.. cpp:function:: template <KokkosView View> \
                  static constexpr auto rank() -> size_t

    :tparam View: The type of the Kokkos view.

    :returns: The rank (number of dimensions) of the ``View`` type.


.. cpp:function:: template <KokkosView View> \
                  static constexpr auto extent(const View &v, const int i) -> size_t

    :tparam View: The type of the Kokkos view.

    :param v: The Kokkos view to query.
    :param i: The index of the dimension. Must be smaller than the ``rank`` of the view.

    :returns: The extent of the specified dimension.


.. cpp:function:: template <KokkosView View> \
                  static constexpr auto stride(const View &v, const int i) -> size_t

    :tparam View: The type of the Kokkos view.

    :param v: The Kokkos view to query.
    :param i: The index of the dimension. Must be smaller than the ``rank`` of the view.

    :returns: The stride of the specified dimension.


.. cpp:function:: template <KokkosView View> \
                  static constexpr auto is_reference_counted() -> bool

    :tparam View: The type of the Kokkos view.

    :returns: True if, and only if, the type is subject to reference counting (e.g., always true for ``Kokkos::View`` objects).

    This is used to determine if asynchronous MPI operations may need to extend the lifetime of this type when it's used as an argument.


Packing Traits
--------------

Strategies for handling non-contiguous views.

.. cpp:namespace:: KokkosComm

.. cpp:struct:: template<typename T> PackTraits<T>

    A common packing-related struct that can be specialized to implement custom behavior for a particular Kokkos view.

    .. cpp:type:: packer_type = Impl::Packer::DeepCopy<View>

    The packer to use for this ``View`` type.

.. .. cpp:function:: static auto needs_unpack(const View &v) -> bool

..     :returns: True if, and only if, the ``v`` needs to be unpacked before being passed to the communication backend.

.. .. cpp:function:: static auto needs_pack(const View &v) -> bool

..     :returns: True if, and only if, the ``v`` needs to be packed before being passed to the communication backend.
