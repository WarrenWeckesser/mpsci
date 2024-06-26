
import sys
import os
import subprocess

from mpsci import fun, stats, distributions


# The function git_version() returns the git revision as a string.
# This function is from scipy's setup.py.  It has the following license:
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - -
# Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - -
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = None

    return git_revision


try:
    os.mkdir('source')
except FileExistsError:
    print("'source' exists.  Either remove or move it.")
    print("No files created.")
    sys.exit(0)


gitrev = git_version()
if gitrev:
    gitrev = gitrev[:8] + "..."
else:
    gitrev = "unknown"

main_descr = """
(*Git revision:* ``%s``)

``mpsci`` is a Python library that defines an assortment of numerical
formulas and algorithms.  The library `mpmath` is used for floating point
calculations.

Most of the code in ``mpsci`` was developed as a way to find the
"true" values to be used in SciPy unit tests.

The package should be considered prototype-quality software.  There
will probably be backwards-incompatible API changes as the code is expanded.

The three subpackages of `mpsci` are:
""" % gitrev

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create README.txt

readme_filename = os.path.join('source', 'README.txt')

with open(readme_filename, 'w') as f:
    f.write('''
DO NOT EDIT ANYTHING IN THIS DIRECTORY!

All the files in this directory (including this one) were generated by
the script '%s'.
''' % __file__)

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
lines.extend(['.. toctree::', ''])
lines.extend(['   fun'])
lines.extend(['   stats'])
lines.extend(['   distributions'])

content = '\n'.join(lines)
index_name = os.path.join('source', 'index.rst')
with open(index_name, 'w') as f:
    f.write(content)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create fun.rst

module = fun
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
    if (src == '_goftests' or src == '_pearsonr' or src == '_anova'
            or src == '_fisher_exact' or src == '_odds_ratio'):
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

names = [name for name in dir(module) if not name.startswith('_')]

objects = [getattr(module, name) for name in names]
continuous_dists = [obj for obj in objects
                    if hasattr(obj, 'pdf')]

univariate_continuous_dists = []
multivariate_continuous_dists = []
for obj in continuous_dists:
    (multivariate_continuous_dists
     if getattr(obj, '_multivariate', False)
     else univariate_continuous_dists).append(obj)

discrete_dists = [obj for obj in objects
                  if hasattr(obj, 'pmf') or hasattr(obj, 'pmf_dict')]

univariate_discrete_dists = []
multivariate_discrete_dists = []
for obj in discrete_dists:
    (multivariate_discrete_dists
     if getattr(obj, '_multivariate', False)
     else univariate_discrete_dists).append(obj)

pth = os.path.join('source', modname)
os.mkdir(pth)

lines.extend(['.. _%s:' % modname, ''])
lines.extend(['.. currentmodule:: mpsci.%s' % modname, ''])
lines.extend(['.. automodule:: mpsci.%s' % modname, ''])

dist_sections = [('Univariate continuous', univariate_continuous_dists),
                 ('Multivariate continuous', multivariate_continuous_dists),
                 ('Univariate discrete', univariate_discrete_dists),
                 ('Multivariate discrete', multivariate_discrete_dists)]
for header, dists in dist_sections:
    # lines.extend(['', '.. currentmodule:: %s' % submodule.__name__, ''])
    lines.extend(['', f'*{header} distributions*', ''])
    lines.extend(['.. autosummary::', ''])
    names = [dist.__name__.split('.')[-1] for dist in dists]
    for name in names:
        lines.extend(['   ' + name])

    for name, dist in zip(names, dists):
        filename = os.path.join(pth, name + '.rst')
        subnames = dist.__all__
        with open(filename, 'w') as f:
            f.write(preamble)
            f.write('\n.. automodule:: mpsci.%s.' % modname + name + '\n')
            f.write('   :members:\n\n')
        print("Created", filename)

lines.extend(['', '*Utility classes*', ''])
lines.extend(['.. autosummary::', ''])
lines.extend(['   Initial'])

content = '\n'.join(lines)
rst = os.path.join('source', modname + '.rst')
with open(rst, 'w') as f:
    f.write(content)

print("Created", rst)
