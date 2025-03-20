# Disk-Based VP-Tree Implementation

This project implements a Vantage-Point Tree (VP-Tree) for nearest neighbor search. It is designed to work with large datasets by performing in-place reordering of embeddings and values stored on disk.  It provides both a command-line interface for building and searching the tree, as well as Python bindings for integration into Python projects.

## Features

*   **Disk-Based:** Handles large datasets that don't fit into memory by using memory-mapped files.
*   **In-Place Reordering:**  Reorders the embedding and value files on disk during tree construction.
*   **Manhattan Distance:** Uses Manhattan (L1) distance for nearest neighbor search.
*   **Half-Precision Support:** Supports both 32-bit floating-point (`float`) and 16-bit floating-point (`half`) embeddings.
*   **Command-Line Interface:** Provides command-line tools for building and searching the VP-Tree.
*   **Python Bindings:** Exposes the VP-Tree functionality to Python using `pybind11`.
*   **Platform-Specific Implementation**: Includes implementation for POSIX systems and Windows, enabling cross-platform compatibility.

## Dependencies

*   C++11 compiler
*   `pybind11` (for Python bindings)
*   Python 3
*   `setuptools`

## Building

### C++ Library and Command-Line Tools

No explicit build steps are needed. The C++ code is primarily designed to be used via the Python bindings. However, the `main` function provides a command-line interface for building and searching the VP-Tree, which can be compiled as follows:

```bash
g++ -std=c++11 diskvec.cpp -o diskvec -D'BUILD_PYBINDINGS=0'
```
