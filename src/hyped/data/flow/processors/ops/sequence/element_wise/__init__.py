"""Module for Element-Wise Operations on Sequences.

This module provides a comprehensive set of processors for performing element-wise operations 
on sequences. It is divided into two primary submodules:

1. **Element-Wise Binary Operations**:
    - This submodule includes processors for binary operations applied element-wise to sequences. 
    - Operations covered include arithmetic (addition, subtraction, multiplication, division), 
      logical (AND, OR, XOR), and comparison (equality, inequality, greater than, less than).
    - The processors handle sequences of numeric, integer, and boolean types, ensuring type 
      consistency and validation. 
    - Key classes: `BaseBinaryElementWiseOp`, `BaseElementWiseComparator`, `BaseElementWiseClosedOp`.

2. **Element-Wise Unary Operations**:
    - This submodule defines processors for unary operations applied element-wise to sequences.
    - Operations include negation, absolute value, bitwise inversion, and boolean inversion.
    - The processors support numeric, integer, and boolean sequence types, transforming data 
      on a per-element basis.
    - Key classes: `BaseUnaryElementWiseOp`, `ElementWiseNeg`, `ElementWiseAbs`, `ElementWiseInvert`.
"""
