# `hyped.extensions` namespace

This directory serves as the **namespace for extensions** to the `hyped` framework. Extensions are optional add-ons that build upon the core functionality of `hyped` and provide specialized nodes for various use cases.

## Overview

The `extensions` namespace is empty by default and acts as a placeholder for external packages that implement additional functionality. These extensions must be installed separately and can then be accessed as part of the `extensions` namespace.

For example, if you install an NLP extension, it will be accessible as:

```python
from hyped.extensions import nlp
```
