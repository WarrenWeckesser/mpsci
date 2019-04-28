"""
    resubber
    ~~~~~~~~

    Simple docstring text subsitution using re.sub.
"""

import re


def process_docstring(app, what, name, obj, options, lines):
    subs = getattr(app.config, 'resubber_re_subs', [])
    subs.extend(getattr(obj, '_docstring_re_subs', []))
    if subs:
        docstring = '\n'.join(lines)
        for pattern, repl, count, flags in subs:
            docstring = re.sub(pattern, repl, docstring,
                               count=count, flags=flags)
        lines.clear()
        lines.extend(docstring.splitlines())


def setup(app):
    app.connect('autodoc-process-docstring', process_docstring)
    app.add_config_value('resubber_re_subs', [], 'env')
    return {
        'version': '0.0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
