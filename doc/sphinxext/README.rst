resubber
--------

This Sphinx extension is a filter for docstrings that makes
substitutions based on regular expressions.

To use it, add the attributes ``_docstring_re_subs`` to any
object whose docstring is to be processed.  The value of
``_docstring_re_subs`` must be a list of tuples.  Each tuple
must contain four elements, ``(pattern, replacement, count, flags)``,
as follows:

pattern
    The pattern in the docstring to be matched and replaced.
replacement
    The string to replace the matched pattern.
count
    Passed on to ``re.sub`` as the ``count`` parameter.
flags
    Passed on to ``re.sub`` as the ``flags`` paramere.


Here's an example from a module in ``mpsci``.  The module
docstring for ``mpsci.distributions.invchi2`` contains the
text:

    The probability density function for the inverse chi-square
    distribution is

        f(x, nu) = 2**(-nu/2) / Gamma(nu/2) * x**(-nu/2 - 1) * exp(-1/(2*x))

    See the Wikipedia article [...]

The line containing ``f(x, nu)`` is fine as a text representation,
but for the online docs, it would be nice to use the Sphinx
``..math::`` directive to render the expression using LaTeX markup
such as::

    .. math::
              f(x, \\nu) = \\frac{2^{-\\nu/2}}{\\Gamma(\\nu/2)}
                           x^{-\\nu/2 - 1} e^{-1/(2x)}


To do this with resubber, we define the variable ``_docstring_re_subs``
in the module, as follows::

    # module docstring substitution
    _math_expression = r"""
    .. math::
              f(x, \\nu) = \\frac{2^{-\\nu/2}}{\\Gamma(\\nu/2)}
                           x^{-\\nu/2 - 1} e^{-1/(2x)}
    """
    _docstring_re_subs = [
        (r'    f\(x,.*$', _math_expression, 0, re.MULTILINE),
        (' nu ', r' :math:`\\nu` ', 0, 0),
    ]

The first pattern to be matched is ``    f\(x,.*$``, which matches the
text starting with ``    f(x`` to the end of the line.  The second value
in ``_docstring_re_subs`` is the replacement text.  The variable
``_math_expression`` has been defined to hold the ``..math::`` directive.

A second substitution is included in ``_docstring_re_subs`` that will
replace occurrences of `` nu  `` with `` :math:`\\nu` ``.  This should
only occur within a line, so the flag ``re.MULTILINE`` is not needed.

For a module, the attribute ``_docstring_re_subs`` is created by
simply assigning it as a variable within the module.  For a function,
the attribute must be explicitly assigned to the function object.

For example, here's a hypothetical module that defines the function
``sincpi``.  The attribute ``_docstring_re_subs`` is assigned as
an attribute of ``sincpi``::

    def sincpi(x):
        """
        Compute sin(pi*x)/(pi*x) with ``sincpi(0)`` defined to be 1.
        """
        [...]

    sincpi._docstring_re_subs = [
        (r'sin\(pi\*x\)/\(pi\*x\)',
         r':math:`\\frac{\\sin(\\pi x)}{\\pi x}`',
         0, 0),
    ]

The substitution will replace ``sin(pi*x)/(pi*x)`` with
``:math:`\\frac{\\sin(\\pi x)}{\\pi x}``.
