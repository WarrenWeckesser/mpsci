
from itertools import groupby
import mpsci.distributions as dists


def is_discrete_dist(dist):
    return hasattr(dist, 'pmf') or hasattr(dist, 'pmf_dict')


def get_distributions():
    names = [name for name in dir(dists)
             if not name.startswith('_') and name != 'Initial']
    objects = [getattr(dists, name) for name in names]
    discrete_flag = [is_discrete_dist(dist) for dist in objects]
    foo = sorted(zip(discrete_flag, objects), key=lambda t: t[0])
    g = groupby(foo, key=lambda t: t[0])
    continuous_dists = [t[1] for t in next(g)[1]]
    discrete_dists = [t[1] for t in next(g)[1]]
    continuous_dists.sort(key=lambda t: t.__name__)
    discrete_dists.sort(key=lambda t: t.__name__)
    return continuous_dists, discrete_dists


def make_column_heading_lines(names):
    maxlen = max(len(name) for name in names)
    padded_names = [f'{name:>{maxlen}s}' for name in names]
    lines = ['  '.join(t) for t in (zip(*padded_names))]
    return lines


def print_impl_table(dists, function_names):
    funcname_col_width = max(len(dist.__name__.split('.')[-1])
                             for dist in dists)
    col_heading_lines = make_column_heading_lines(function_names)
    for line in col_heading_lines:
        print(f'{" ":{funcname_col_width}s} {line}')
    for dist in dists:
        print(f'{dist.__name__.split(".")[-1]:{funcname_col_width}s} ', end='')
        for funcname in function_names:
            flag = hasattr(dist, funcname)
            print(f'{"✔" if flag else "-":3s}', end='')
        print()


cont_dists, disc_dists = get_distributions()
cont_function_names = ['pdf', 'logpdf',
                       'interval_prob',
                       'cdf', 'logcdf', 'invcdf', 'sf', 'logsf', 'invsf',
                       'mode', 'median',
                       'mean', 'var', 'skewness', 'kurtosis', 'entropy',
                       'noncentral_moment',
                       'nll', 'mle', 'mom']
disc_function_names = ['pmf', 'pmf_dict', 'logpmf', 'logpmf_dict', 'support',
                       'cdf', 'invcdf', 'sf', 'invsf',
                       'mode', 'median',
                       'mean', 'var', 'cov', 'skewness', 'kurtosis', 'entropy',
                       'nll', 'mle', 'mom']

print_impl_table(cont_dists, cont_function_names)
print()
print_impl_table(disc_dists, disc_function_names)
