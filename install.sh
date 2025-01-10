rm -rf build 2> /dev/null
rm -rf dist 2> /dev/null
rm -rf *.egg-info 2> /dev/null


pip install -r requirements.txt
python3 setup.py build_ext 
python3 setup.py bdist_wheel
pip install --force-reinstall dist/*.whl
