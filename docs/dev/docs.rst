***************************
Extending the documentation
***************************

Using reStructedText
====================

Useful resources:

* `Basics of rST <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
* `Documenting C++ with rST <https://www.sphinx-doc.org/en/master/usage/domains/cpp.html>`_


Building a local copy of the docs
=================================

1. Create a Python virtual environment at ``.venv``:

    .. code-block:: console

        $ python3 -m venv .venv

2. Activate the virtual environment:

    .. code-block:: console

        $ source .venv/bin/activate

3. Install the documentaion pre-requisites:

    .. code-block:: console

        $ pip install -r docs/requirements.txt

4. Build the documentation:

    .. code-block:: console

        $ make -C docs html

5. Open the documentation in your favorite browser:

    .. code-block:: console

        $ <BROWSER> docs/_build/html/index.html
