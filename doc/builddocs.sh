# Remove all the old files.
rm -rf source
rm -rf build

# Generate the .rst files in source/
python myautogen.py

# Run sphinx to build the html pages.
sphinx-build -c . -b html -a -E source build

# Remove sphinx intermediate files.
rm -f build/.buildinfo
rm -rf build/.doctrees
