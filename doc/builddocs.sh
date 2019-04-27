rm -rf source
rm -rf build
python myautogen.py
sphinx-build -c . -b html -a -E source build
