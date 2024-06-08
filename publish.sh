##!/bin/bash
rm -rf dist
python setup.py sdist bdist_wheel --universal
twine upload --verbose dist/* --config-file ~/.pypirc