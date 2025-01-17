************
Contributing
************

Workflow
========

Contributions are accepted as pull requests (PR) against the ``develop`` branch on `GitHub <https://github.com/kokkos/kokkos-comm/pulls>`_.
Unless there are extenuating circumstances, PRs must pass all automatic tests before being merged, and will require approvals from core team members.
Please freely use GitHub Issues/Discussions, the Kokkos Team Slack, or emails to discuss any proposed contributions.

In very limited circumstances, modifications may need to be tested as branches in the repository rather than pull requests.
Such changes should always be made in consultation with the core development team.

For a more detailed overview of what is expected of a "good" PR, please refer to `this section <https://kokkos.org/kokkos-core-wiki/developer-guides/prs-and-reviews.html>`_ of the Kokkos documentation.


Code formatting
===============

All code shall be formatted with clang-format 14:

.. code-block:: console

    $ shopt -s globstar
    $ clang-format-14 -i {src,unit_tests,perf_tests}/**/*.[ch]pp


Alternatively, you can use Docker/Podman:

.. code-block:: console

    $ shopt -s globstar
    $ podman run --rm -v ${PWD}:/src ghcr.io/cwpearson/clang-format-14 clang-format -i {src,unit_tests,perf_tests}/**/*.[ch]pp

.. important:: The above command expects ``$PWD`` to be the KokkosComm tree.


Site-Specific Documentation
===========================

These sites may be non-public and usually require credentials or affiliation at the institution to access:

* `Sandia National Laboratories <https://gitlab-ex.sandia.gov/cwpears/kokkos-comm-internal/-/wikis/home>`_


Behavioral Expectations
=======================

Those who are unwilling or unable to collaborate in a respectful manner, regardless of time, place, or medium, are expected to redirect their effort and attention to other projects.
