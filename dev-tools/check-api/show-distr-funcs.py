
from itertools import groupby
import mpsci.distributions as dists


def is_discrete_dist(dist):
    return hasattr(dist, 'pmf') or hasattr(dist, 'pmf_dict')


def split_mv(dists):
    uni = []
    multi = []
    for dist in dists:
        mvflag = getattr(dist, '_multivariate', False)
        (multi if mvflag else uni).append(dist)
    return uni, multi


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
    continuous_uni, continuous_multi = split_mv(continuous_dists)
    discrete_uni, discrete_multi = split_mv(discrete_dists)
    return continuous_uni, continuous_multi, discrete_uni, discrete_multi


def column_heading_subs(names):
    names = names.copy()
    subs = []
    for k in range(len(names)):
        name = names[k]
        if len(name) > 8:
            subs.append(name)
            names[k] = str(len(subs))
    return names, subs


def make_column_heading_lines(names):
    maxlen = max(len(name) for name in names)
    padded_names = [f'{name:>{maxlen}s}' for name in names]
    lines = ['  '.join(t) for t in (zip(*padded_names))]
    return lines


def print_impl_table(title, dists, function_names):
    print()
    print(title)
    print('='*len(title))
    print()
    function_names, subs = column_heading_subs(function_names)
    funcname_col_width = max(len(dist.__name__.split('.')[-1])
                             for dist in dists)
    col_heading_lines = make_column_heading_lines(function_names)
    for k in range(len(subs)):
        print(f'{k+1} = {subs[k]}')
    print()
    for line in col_heading_lines:
        print(f'{" ":{funcname_col_width}s} {line}')
    for dist in dists:
        print(f'{dist.__name__.split(".")[-1]:{funcname_col_width}s} ', end='')
        for funcname in function_names:
            flag = hasattr(dist, funcname)
            print(f'{"✔" if flag else "-":3s}', end='')
        print()


cont_uni, cont_multi, disc_uni, disc_multi = get_distributions()
cont_uni_function_names = ['pdf', 'logpdf',
                           'interval_prob',
                           'cdf', 'logcdf', 'invcdf', 'sf', 'logsf', 'invsf',
                           'mode', 'median',
                           'mean', 'var', 'skewness', 'kurtosis', 'entropy',
                           'noncentral_moment',
                           'nll', 'mle', 'mom']
cont_multi_function_names = ['pdf', 'logpdf',
                             'cdf', 'logcdf', 'invcdf', 'sf', 'logsf', 'invsf',
                             'mode', 'median',
                             'mean', 'var', 'cov', 'entropy',
                             'noncentral_moment',
                             'nll', 'mle', 'mom']
disc_uni_function_names = ['pmf', 'logpmf',
                           'support', 'support_pmf',
                           'cdf', 'invcdf', 'sf', 'invsf',
                           'mode', 'median',
                           'mean', 'var', 'skewness', 'kurtosis', 'entropy',
                           'nll', 'mle', 'mom']
disc_multi_function_names = ['pmf', 'logpmf',
                             'support', 'support_pmf',
                             'cdf', 'invcdf', 'sf', 'invsf',
                             'mode', 'median',
                             'mean', 'var', 'cov', 'entropy',
                             'nll', 'mle', 'mom']

print_impl_table('Continuous univarate distributions',
                 cont_uni, cont_uni_function_names)
print()
print_impl_table('Continuous multivariate distributions',
                 cont_multi, cont_multi_function_names)
print()
print_impl_table('Discrete univariate distributions',
                 disc_uni, disc_uni_function_names)
print()
print_impl_table('Discrete multivariate distributions',
                 disc_multi, disc_multi_function_names)
