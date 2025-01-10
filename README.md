# CLEARBOX ENGINE

## Installation

```shell
$ pip install -r requirements.txt
$ python setup.py build_ext 
$ python setup.py bdist_wheel
$ pip install --force-reinstall dist/*.whl
```

# Try it out
Take a quick look at how the generation and evaluation process with Clearbox Engine works.
## !!!! Verifica che il link Colab funzioni !!!!
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clearbox-AI/engine/blob/main/examples/tabular_data_generation.ipynb)

Find other examples [here](https://github.com/Clearbox-AI/engine/examples).



## Obfuscation

### Reference
* [Using Cython to protect a Python codebase ](https://bucharjan.cz/blog/using-cython-to-protect-a-python-codebase.html)
* [Package only binary compiled so files of a python library compiled with Cython](https://stackoverflow.com/questions/39499453/package-only-binary-compiled-so-files-of-a-python-library-compiled-with-cython)
* [Distributing python packages protected with Cython](https://medium.com/swlh/distributing-python-packages-protected-with-cython-40fc29d84caf)
