#!/bin/bash

set -e

cd /io

${PYTHON_PATH}/pip install -U pip setuptools wheel Cython
${PYTHON_PATH}/pip install -r requirements.txt
${PYTHON_PATH}/python setup.py bdist_wheel

# Repair the wheels with auditwheel
for whl in dist/*.whl; do
    auditwheel repair "$whl" --plat manylinux2014_x86_64 -w /io/wheelhouse/

rm -rf dist
mv wheelhouse dist
done
