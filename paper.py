import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import metric
import strings as s
from common import vanilla_columns
from experiments.management.experiment_manager import SyntheticExperimentManager, RealExperimentManager, \
    BaseSyntheticExperimentManager
from experiments.result_parsing.ff_sparsity_plot import sparsity_plot, sparsity_hist, fontsize
from metric import Metric

# Managers for the synthetic/real experiments
synthetic_experiment_manager = SyntheticExperimentManager(name='experiment_1_synthetic_data', metric=s.ALL_METRICS)
real_experiment_manager = RealExperimentManager(name='experiment_2_real_data', metric=[s.EUCLIDEAN, s.MANHATTAN])

# Managers for the sparsity experiments
dsn_sparsity = [f'{s.SPARSE_BLOB}_10000_{nf}_{nc}_0_{i:.2f}' for i, nc, nf in
                itertools.product(np.arange(0.01, 1.01, 0.01), range(5, 31, 5), np.arange(4, 65, 4))]
mics_sparsity = filter(lambda x: x != metric.AQBC, metric.Metric.metric_implementations(s.EUCLIDEAN))
sparsity_em_voronoi = BaseSyntheticExperimentManager(dsn_list=dsn_sparsity, metric=s.EUCLIDEAN,
                                                     name=f'experiment_3_sparsity',
                                                     mics=mics_sparsity)

sparsity_em_distance = BaseSyntheticExperimentManager(dsn_list=dsn_sparsity, metric=s.EUCLIDEAN,
                                                      name=f'experiment_3_sparsity',
                                                      mics=[metric.EUCLIDEAN_VANILLA])

try:
    from local import fd_paper
except ImportError:
    fd_paper = './paper_results'

fd_paper = Path(fd_paper)
fd_paper.mkdir(exist_ok=True, parents=True)


def figure_rationale():
    # Figure 1
    f, ax = plt.subplots()

    p_data = pd.DataFrame(
        data={'x': [.85, .9, 1.05, 0.1],
              'y': [1.1, .1, 0.05, 0.05]},
        index=list('abcd')
    )

    d_data = pd.DataFrame(
        data={'x': [1, 1, 0, 0],
              'y': [1, 0, 0, 1]},
        index=list('efgh')
    )

    color_p = (.7,) * 3
    color_d = 'k'
    ms = 10
    offset = .03

    ax.plot(p_data.x, p_data.y, linestyle='', marker='o', markersize=ms, color=color_p)
    ax.plot(d_data.x, d_data.y, linestyle='', marker='o', markersize=ms, color=color_d)
    ax.plot(p_data.loc[list('dab')].x, p_data.loc[list('dab')].y, '--', color=color_p)
    ax.plot(d_data.loc[list('gef')].x, d_data.loc[list('gef')].y, '--', color=color_d)

    for df, col in zip([p_data, d_data], [color_p, color_d]):
        for letter, coordinates in df.iterrows():
            ax.text(x=coordinates['x'] + offset,
                    y=coordinates['y'] - offset,
                    color=col, s=str(letter), ha='left', va='center',
                    fontdict=dict(fontsize=12, fontweight='bold'))

    ax.set_aspect('equal')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.tick_params(width=2)

    ax.set_xticklabels(['0', '1'], fontdict=dict(fontsize=12, fontweight='bold'))
    ax.set_yticklabels(['0', '1'], fontdict=dict(fontsize=12, fontweight='bold'))
    f.set_size_inches(4, 4)

    plt.savefig(fd_paper / 'rationale.eps', bbox_inches='tight')
    plt.close()


def figure_issue():
    # Figure 2
    f, ax = plt.subplots()

    # POINTS
    dx = 5
    ddx = 0.2
    points_y = 0.3
    ld = 0.2

    x_pos = [-dx + ddx, -ddx, ddx, dx - ddx]
    ax.plot(x_pos, [points_y] * 4, marker='o', linestyle='', ms=5, color='k')
    for x, l in zip(x_pos, list('cabd')):
        ax.text(x + (ddx if l in 'bc' else -ddx), points_y, l, va='center', ha='center',
                fontsize=15)

    for x in [-dx, 0, dx]:
        plt.plot([x] * 2, [points_y - ld, points_y + ld], linestyle='--', color='k')
    ax.text(x=-dx / 2, y=points_y + ld, s='$f_j^n(x)=t$', va='center', ha='center', fontsize=15)
    ax.text(x=dx / 2, y=points_y + ld, s='$f_j^n(x)=t+1$', va='center', ha='center', fontsize=15)

    t = np.linspace(-10, 10, 100)

    for a in [+1, -1]:
        x = dx / 2 / (1 + np.exp(t)) + a * dx / 2
        y = points_y + .1 + t / 120
        ax.plot(x, y, 'k')

        x = -dx / 2 / (1 + np.exp(t)) + a * dx / 2
        y = points_y + .1 + t / 120
        ax.plot(x, y, 'k')

    ax.set_ylim(-.3, .7)
    f.set_size_inches(8, 4)
    ax.get_xaxis().set_visible(False)
    plt.axis('off')
    plt.savefig(fd_paper / 'discretizationissue.eps', bbox_inches='tight')
    plt.close()


def make_pure():
    # Figure 3
    from experiments.result_parsing.ee_pure import pure_plot
    f, axarr = plt.subplots(1, 4)

    log_ys = [s.DURATION, s.PREPROCESSING_DURATION]

    for ax, yf in zip(axarr.flatten(),
                      [s.PREPROCESSING_DURATION, s.DURATION, s.MAX_RELATIVE_ERROR, s.MEAN_RELATIVE_ERROR]):
        pure_plot(alm=synthetic_experiment_manager, y_axis=yf, ax=ax, number_of_datapoints=10000,
                  legend=yf == s.MAX_RELATIVE_ERROR,
                  skip_mic=Metric.metric_implementations([s.WEIGHTED_JACCARD]) + [metric.AQBC],
                  logy=yf in log_ys,
                  mult_y=1e6 if yf in [s.PREPROCESSING_DURATION] else (1e9 if yf in [s.DURATION] else 1),
                  err_base='respective')

    # preprocessing ~ 1E-5
    axarr.flatten()[0].set_title(r'Preprocessing ($\mu$s)', fontsize=15)
    from matplotlib import ticker
    axarr.flatten()[0].yaxis.set_major_formatter(ticker.ScalarFormatter())
    axarr.flatten()[0].set_yticks([0.1, 1, 10, 100])
    axarr.flatten()[0].set_yticklabels(['0.1', '1', '10', '100'])
    # axarr.flatten()[0].set_yticks(np.arange(0, 13, 2) * 10 ** -6)
    # axarr.flatten()[0].set_yticklabels([f'{i}' for i in range(0, 13, 2)])

    # computation is logarithmic
    axarr.flatten()[1].set_title('Computation (ns)', fontsize=15)
    axarr.flatten()[1].yaxis.set_major_formatter(ticker.ScalarFormatter())

    for ax in axarr.flatten()[2:]:
        ax.set_title(ax.get_ylabel(), fontsize=15)

    for ax in axarr.flatten():
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel('$|F|$', fontsize=15)
        ax.set_ylabel('')
        ax.set_xticks(range(4, 65, 20))
        ax.set_xticklabels(['4', '24', '44', '64'])

    f.set_size_inches(w=20, h=5)
    plt.savefig(fd_paper / 'pure_experiments.pdf', bbox_inches='tight')
    plt.savefig(fd_paper / 'pure_experiments.png', bbox_inches='tight')
    plt.close()


bb_mics = sum([[f'{d}BB{i}' for i in range(1, 4)] for d in 'EM'], [])

other_lloyd_mics = [metric.JKKC, metric.AMAX_BMIN, metric.MANHATTAN_VANILLA, metric.WEIGHTED_JACCARD_VANILLA] \
                   + [f'{metric.WEIGHTED_JACCARD_MIN_HASH}{i}' for i in [4, 9]] + [metric.AQBC, metric.BITBOOSTED_AQBC]

lloyd_mics = bb_mics + other_lloyd_mics

lloyd_mics2 = bb_mics + [metric.JKKC, metric.BITBOOSTED_AQBC, metric.MANHATTAN_VANILLA]
real_mics = lloyd_mics2 + [metric.EUCLIDEAN_VANILLA, metric.AMAX_BMIN]
real_mics2 = bb_mics[:3] + [metric.AMAX_BMIN, metric.JKKC, metric.BITBOOSTED_AQBC] + [metric.EUCLIDEAN_VANILLA]


def table_iterations():
    # Table 4
    df = synthetic_experiment_manager.import_parsed(s.VORONOI)
    df = df.groupby(s.METRIC_IMPLEMENTATION_CODE)[s.ITERATION_COUNT].mean().to_frame()
    df = df.loc[lloyd_mics2 + [metric.EUCLIDEAN_VANILLA], :].reset_index()
    df[s.METRIC_IMPLEMENTATION_CODE] = df[s.METRIC_IMPLEMENTATION_CODE].apply(lambda x: Metric(x).short_tex)
    df[s.ITERATION_COUNT] = df[s.ITERATION_COUNT].apply(lambda x: f'${x:.2f}$')

    cutoff = (len(df) + 1) // 2
    df_top = df.iloc[:cutoff].reset_index(drop=True)
    df_bottom = df.iloc[cutoff:].reset_index(drop=True)
    df = pd.concat([df_top, df_bottom], axis=1).fillna('')
    # TODO sort on mic
    foo = df.rename(columns={s.METRIC_IMPLEMENTATION_CODE: 'Implementation', s.ITERATION_COUNT: 'Iterations'}) \
        .to_latex(index=False, escape=False, column_format='{p{2cm}r|p{2cm}r}')
    with open(fd_paper / 'iterations.tex', 'w+') as wf:
        wf.write(foo)


def _pareto(synth, **kwargs):
    # Figure 4
    from experiments.result_parsing.cc_pareto import make_xfa
    single = False
    use_ari = True

    f, axarr = make_xfa(xfa=s.DURATION, by=s.METRIC_IMPLEMENTATION_CODE, ca=s.VORONOI,
                        yfa=s.ARI if use_ari else s.PURITY_WITHOUT_NOISE, do_mics=kwargs.get('do_mics', lloyd_mics),
                        single=single, alm=synthetic_experiment_manager if synth else real_experiment_manager, cols=3)

    for ax in axarr[-1, :].flatten():
        ax.set_xlabel('Relative duration')
    for ax in axarr[:, 0].flatten():
        if use_ari:
            ax.set_ylabel('Relative ARI')
        else:
            ax.set_ylabel('Relative purity')
    for ax in axarr[:, 1:].flatten():
        ax.set_ylabel('')
    for ax in axarr[:-1, :].flatten():
        ax.set_xlabel('')

    if synth:
        nr, nc = f.axes[0].get_subplotspec().get_gridspec().get_geometry()
        f.set_size_inches(h=nr * 2.5, w=nc * 5)

        for ax in axarr.flatten():
            t = ax.get_title()
            ax.set_title('')
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.text(0.5 * (xmax - xmin) + xmin, 0.75 * (ymax - ymin) + ymin, t, ha='center',
                    fontdict=dict(size=20))

    if single:
        axarr.flatten()[0].set_xlim(0, 5)

    plt.savefig(fd_paper / f'{"synthetic" if synth else "real"}_pareto.pdf', bbox_inches='tight')
    plt.close()


def real_pareto():
    _pareto(synth=False)


def synthetic_pareto(**kwargs):
    _pareto(synth=True, **kwargs)


default_mics = Metric.metric_implementations(s.EUCLIDEAN) + \
               Metric.metric_implementations(s.MANHATTAN) + \
               lloyd_mics[-2:]


def table_with_real_results(ca, mics=default_mics):
    # Table 5
    # real_experiment_manager.metric = s.EUCLIDEAN
    if not real_experiment_manager.fn_parsed(ca).exists():
        real_experiment_manager.parse_results(ca)
    from experiments.result_parsing.bb_tex import do_single_alt
    qfs = vanilla_columns(True, ca) + [s.ARI, s.PURITY_WITHOUT_NOISE]
    for qf in qfs:
        kwargs = dict(
            alm=real_experiment_manager,
            ca=ca,
            quality_feature=qf,
            duration_feature=s.DURATION,
            add_ratio=False,
            used_mics=mics,
            to_clip=False,
            sparsity_metrics_shown=s.EUCLIDEAN,
            sparsity_mode=s.AVG_DISTANCE,
        )

        latex_str = do_single_alt(skip_dsn=['vicon1', 'iris'], **kwargs)

        with open(fd_paper / f'results_real_{ca}_{qf}.tex', 'w+') as wf:
            wf.write(latex_str)
        try:
            from functions import tex_functions
            x = do_single_alt(wrap_as_table=False, **kwargs)
            standalone = tex_functions.tabular_to_standalone_tex(x)
            fn_tex = Path('./temp/temp.tex')
            fn_tex.parent.mkdir(exist_ok=True, parents=True)
            with open(fn_tex, 'w+') as wf:
                wf.write(standalone)
            tex_functions.tex_2_svg(fn_tex, fd_paper / f'results_real_{ca}_{qf}.svg')

        except ModuleNotFoundError:
            pass


def appendix_truth_table():
    # Tables 6 + 7
    from bitbooster.other.karnaugh import euclidean, generate_dfs
    foo = generate_dfs(3, euclidean)
    for i in [3, 4]:
        foo[i].index.name = rf'$z[{i}]$'

        latex_string = '\\begin{table}\n'
        latex_string += f'\\caption{{Truth table for $z[{i}]$}}\n'
        latex_string += f'\\label{{tab:tt{i}}}\n'

        lines = foo[i].reset_index().to_latex(column_format='l|' + 'r' * 4 + '|' + 'r' * 4,
                                              escape=False, index=False)
        latex_string += '\n'.join(lines.split('\n')[:8])
        latex_string += '\n\\hline\n'
        latex_string += '\n'.join(lines.split('\n')[8:])

        latex_string += '\\end{table}'

        with open(fd_paper / f'truth_table{i}.tex', 'w+') as wf:
            wf.write(latex_string)


def sparsity_results():
    sparsity_kwargs = dict(binning_type='maximize_n_bins_all_points', binning_parameter=30,
                           x_feature=s.AVG_4DIST, metric=s.EUCLIDEAN, blm=sparsity_em_voronoi)
    sparsity_em_voronoi.parse_results(s.VORONOI)

    kwargs = sparsity_kwargs.copy()

    # Axes to create figure in
    f, ax = plt.subplots()

    # Make two plots
    ax2 = ax.twinx()

    # Lines with results
    sparsity_plot(ca=s.VORONOI, ax=ax, plot_dict=dict(ls='-'), **sparsity_kwargs)

    # Histogram with number of datapoints
    sparsity_hist(plot_dict=dict(alpha=0.1), normalize=True, ax=ax2, **kwargs)

    # Fix labels
    ax.set_ylabel('')
    ax2.set_ylabel('')
    ax.set_xlabel('')
    f.text(0.07, 0.5, 'Fraction', va='center', rotation='vertical', size=fontsize)
    f.text(0.92, 0.5, 'Subset size', va='center', rotation='vertical', size=fontsize)
    f.text(0.5, 0.02, 'Sparsity $s_E$', ha='center', size=fontsize)

    # Save figure
    f.set_size_inches(w=6, h=3)
    plt.savefig('paper_results/syntheticsparsity.pdf', bbox_inches='tight')


if __name__ == '__main__':
    # Pure experiment
    make_pure()

    # Synthetic Voronoi
    synthetic_experiment_manager.parse_results(s.VORONOI)
    table_iterations()
    synthetic_pareto(do_mics=lloyd_mics2)

    # Real DBSCAN
    real_experiment_manager.run_vanilla_metrics(s.DBSCAN)
    real_experiment_manager.parse_results(s.DBSCAN)
    table_with_real_results(s.DBSCAN, mics=real_mics)

    # Real Voronoi
    real_experiment_manager.parse_results(s.VORONOI)
    table_with_real_results(s.VORONOI, mics=real_mics2)

    # Synthetic Sparsity
    sparsity_results()

    # Appendix
    appendix_truth_table()
