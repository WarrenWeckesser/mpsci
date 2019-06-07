
import sys
import os
from mpsci import fun, signal, stats, distributions, polyapprox


try:
    os.mkdir('source')
except FileExistsError:
    print("'source' exists.  Either remove or move it.")
    print("No files created.")
    sys.exit(0)

main_descr = """
``mpsci`` is a Python module that defines an assortment of numerical
formulas and algorithms.  The library `mpmath` is used for floating point
calculations.

Most of the code in ``mpsci`` was developed as a way to find the
"true" values to be used in SciPy unit tests.

The package should be considered prototype-quality software.  There
will probably be backwards-incompatible API changes as the code is expanded.

The five subpackages of `mpsci` are:
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create README.txt

readme_filename = os.path.join('source', 'README.txt')

with open(readme_filename, 'w') as f:
    f.write('''
DO NOT EDIT ANYTHING IN THIS DIRECTORY!

All the files in this directory (including this one) were generated by
the script '%s'.
''' % __file__)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

preamble_lines = [
    '..',
    '   DO NOT EDIT THIS FILE!',
    '   ',
    "   This file was generated by the script '%s'." % __file__,
    '',
    '',
]

preamble = '\n'.join(preamble_lines)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create index.rst

lines = []
lines.extend(preamble_lines)
lines.extend(['mpsci', '=====', ''])
lines.extend([main_descr, ''])
lines.extend(['.. toctree::',
              '   :maxdepth: 1',
              '   :titlesonly:',
              ''])
lines.extend(['   fun', ''])
lines.extend(['   signal', ''])
lines.extend(['   stats', ''])
lines.extend(['   distributions', ''])
lines.extend(['   polyapprox', ''])

content = '\n'.join(lines)
index_name = os.path.join('source', 'index.rst')
with open(index_name, 'w') as f:
    f.write(content)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create fun.rst, signal.rst and polyapprox.rst

for module in [fun, signal, polyapprox]:
    modname = module.__name__.split('.')[-1]
    names = [name for name in dir(module) if not name.startswith('_')]

    lines = []
    lines.extend(preamble_lines)
    lines.extend(['.. _%s:' % modname, ''])
    lines.extend(['.. currentmodule:: mpsci.%s' % modname, ''])
    lines.extend(['.. automodule:: mpsci.%s' % modname, ''])
    lines.extend(['.. autosummary::',
                  ''])
    for name in names:
        lines.extend(['   ' + name])

    content = '\n'.join(lines)
    rst = os.path.join('source', modname + '.rst')
    with open(rst, 'w') as f:
        f.write(content)

    print("Created", rst)

    pth = os.path.join('source', modname)
    os.mkdir(pth)
    for name in names:
        filename = os.path.join(pth, name + '.rst')
        with open(filename, 'w') as f:
            f.write(preamble)
            f.write('.. _' + '_'.join(['mpsci', modname, name]) + ':\n')
            f.write('\n%s\n' % name)
            f.write('-'*len(name) + '\n\n')
            f.write('\n.. autofunction:: mpsci.%s.' % modname + name + '\n')
        print("Created", filename)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create stats.rst

module = stats
modname = 'stats'
names = [name for name in dir(module) if not name.startswith('_')]

testnames = []
othernames = []
for name in names:
    obj = getattr(stats, name)
    src = obj.__module__.split('.')[-1]
    if src == '_goftests' or src == '_pearsonr' or src == '_fisher_exact':
        testnames.append(name)
    else:
        othernames.append(name)

lines = []
lines.extend(preamble_lines)
lines.extend(['.. _%s:' % modname, ''])
lines.extend(['.. currentmodule:: mpsci.%s' % modname, ''])
lines.extend(['.. automodule:: mpsci.%s' % modname, ''])

for names, title in [(othernames, 'Descriptive statistics'),
                     (testnames, 'Statistical tests')]:
    lines.extend(['*' + title + '*', ''])
    lines.extend(['.. autosummary::', ''])
    for name in names:
        lines.extend(['   ' + name])
    lines.append('')

content = '\n'.join(lines)
rst = os.path.join('source', modname + '.rst')
with open(rst, 'w') as f:
    f.write(content)

print("Created", rst)

pth = os.path.join('source', modname)
os.mkdir(pth)
for name in othernames + testnames:
    filename = os.path.join(pth, name + '.rst')
    with open(filename, 'w') as f:
        f.write(preamble)
        f.write('\n%s\n' % name)
        f.write('-'*len(name) + '\n\n')
        f.write('\n.. autofunction:: mpsci.%s.' % modname + name + '\n')
    print("Created", filename)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create distributions.rst


lines = []
lines.extend(preamble_lines)

module = distributions
modname = module.__name__.split('.')[-1]

pth = os.path.join('source', modname)
os.mkdir(pth)

lines.extend(['.. _%s:' % modname, ''])
lines.extend(['.. currentmodule:: mpsci.%s' % modname, ''])
lines.extend(['.. automodule:: mpsci.%s' % modname, ''])

for submodule in [distributions.continuous, distributions.discrete]:
    submodname = submodule.__name__.split('.')[-1]
    #lines.extend(['', '.. currentmodule:: %s' % submodule.__name__, ''])
    lines.extend(['', '*' + submodname.title() + ' distributions*', ''])
    names = [name for name in dir(submodule) if not name.startswith('_')]
    lines.extend(['.. autosummary::', ''])
    for name in names:
        lines.extend(['   ' + name])

    for name in names:
        filename = os.path.join(pth, name + '.rst')
        submodule = getattr(module, name)
        subnames = submodule.__all__
        with open(filename, 'w') as f:
            f.write(preamble)
            f.write('\n.. automodule:: mpsci.%s.' % modname + name + '\n')
            f.write('   :members:\n\n')
        print("Created", filename)


content = '\n'.join(lines)
rst = os.path.join('source', modname + '.rst')
with open(rst, 'w') as f:
    f.write(content)

print("Created", rst)
