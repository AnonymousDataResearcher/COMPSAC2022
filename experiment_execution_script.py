# Synthetic ------------------------------------------------------------------------------------------------------------
import sys

from paper import real_experiment_manager, synthetic_experiment_manager, sparsity_em_distance, sparsity_em_voronoi


# Distance Experiments -------------------------------------------------------------------------------------------------
def distance_experiment():
    synthetic_experiment_manager.run_pure_and_4dist()


# Synthetic Experiments ------------------------------------------------------------------------------------------------
def synthetic_experiment():
    synthetic_experiment_manager.run_voronoi()


# Real -----------------------------------------------------------------------------------------------------------------
# This prepares for dbscan eps estimation (Automatic)
def pure_real():
    real_experiment_manager.run_pure_and_4dist()


# This is the dbscan eps estimation (Manual Input required)
def do_eps_estimate():
    real_experiment_manager.run_eps_estimation()


# Do all dbscan experiments
def real_dbscan():
    real_experiment_manager.run_dbscan()


def real_voronoi():
    real_experiment_manager.run_voronoi()


# This is all three parts for real
def real_full():
    pure_real()
    do_eps_estimate()
    real_dbscan()
    real_voronoi()


def real_skip_eps():
    pure_real()
    from shutil import copyfile
    copyfile(src='_precomputed/experiment_2_real_data/4dist/_eps.csv',
             dst='real/experiment_2_real_data/_eps.csv')
    print('used eps from the paper')
    real_dbscan()


def sparsity_experiments():
    sparsity_em_distance.run_pure_and_4dist()
    sparsity_em_voronoi.run_voronoi()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        distance_experiment()
        synthetic_experiment()
        real_skip_eps()
        sparsity_experiments()
    elif sys.argv[1] == 'synthetic_experiment':
        synthetic_experiment()
    elif sys.argv[1] == 'distance_experiment':
        distance_experiment()
    elif sys.argv[1] == 'real_experiment' and len(sys.argv) == 3 and sys.argv[2] == 'skip_eps':
        real_skip_eps()
    elif sys.argv[1] == 'real_experiment' and len(sys.argv) == 2:
        real_full()
    elif sys.argv[1] == 'sparsity_experiments':
        sparsity_experiments()
    else:
        print('invalid parameters. Use one of:')
        print('<none>                       For all experiments')
        print('distance_experiment          For experiment 1')
        print('synthetic_experiment         For experiment 2')
        print('real_experiment              For experiment 3')
        print('real_experiment skip_eps     For experiment 3 with saved eps values')
        print('sparsity_experiments         For experiment 3 with saved eps values')
