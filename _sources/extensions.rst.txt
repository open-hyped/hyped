Extensions
==========

Extend the functionality of `hyped` by installing extensions, which provide additional features and capabilities to the framework. Extensions are available as separate packages that can be installed via pip and then imported as sub-packages within your code.

Installing Extensions
---------------------

You can install `hyped` extensions using pip:

.. code-block:: bash

    pip install hyped-extensions-nlp

Once installed, you can import add-ons as sub-packages within your code:

.. code-block:: python

    import hyped.extensions.nlp

Available Extensions
--------------------

Here is a list of available extensions. Please note that extensions are developed by the community as well, more may be added in the future:

- `hyped-extensions-nlp <https://github.com/open-hyped/hyped-extensions-nlp>`_: This extension provides Natural Language Processing (NLP) tools and features for the :code:`hyped` framework. It includes utilities for text preprocessing, tokenization and other NLP tasks, helping users to easily integrate NLP functionality into their workflows.
- `hyped-extensions-parsers <https://github.com/open-hyped/hyped-extensions-parsers>`_: This extension offers parsers for various data formats, enabling users to easily parse different input types (e.g., JSON, CSV, etc.) within the :code:`hyped` framework.

Implementing an Extension
-------------------------

To implement a custom extension, please refer to the `Coockiecutter project <https://github.com/open-hyped/hyped-extensions-cookiecutter>`_ for a template and guidelines on creating your own extension.