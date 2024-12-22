sphinx-apidoc -f -e -o source/api ../src --maxdepth 3
make clean
make html
