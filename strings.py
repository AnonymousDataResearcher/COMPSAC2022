from bitbooster.strings import *

# Experiment status
STATUS = 'status'
COMPLETED = 'CP'
IMPOSSIBLE_PARAMETER_COMBINATION = 'IM'
UNLUCKY_SEED = 'SD'
LLOYD_FAIL = 'LF'
NOT_COMPLETED = 'NC'
JKKC_ZERO_VECTOR = 'J0V'
AQBC_NEGATIVE_ERROR = 'ANE'

# DATASET PROPERTIES ---------------------------------------------------------------------------------------------------
DATASET = 'dataset'
SPARSITY = 'sparsity'
PROPERTY = 'property'
VALUE = 'value'
LABEL = 'label'
FEATURE = 'feature'
NUMBER_OF_CLASSES = 'number_of_classes'
ANNOTATED = 'annotated'
NUMBER_OF_DATAPOINTS = 'n_datapoints'
NUMBER_OF_CLUSTERS = 'n_clusters'
NUMBER_OF_FEATURES = 'n_features'
RADIUS = 'radius'
SEED = 'seed'
dataset_features = [SEED, NUMBER_OF_DATAPOINTS, NUMBER_OF_FEATURES, NUMBER_OF_CLUSTERS]

# SYNTHETIC DATASET TYPES-----------------------------------------------------------------------------------------------
BLOB = 'blob'
SPARSE_BLOB = 'sblob'
SHAPE = 'shape'
all_shapes = [BLOB]

# METRIC_IMPLEMENTATION-------------------------------------------------------------------------------------------------
METRIC_IMPLEMENTATION_CODE = 'metric_implementation_code'

# EXECUTION STATISTICS--------------------------------------------------------------------------------------------------
PREP_MEM_PEAK = f'preprocessing_memory_peak'
PREP_MEM_MEAN = f'preprocessing_memory_mean'
COMP_MEM_PEAK = f'computation_memory_peak'
COMP_MEM_MEAN = f'computation_memory_mean'
ITERATION_COUNT = 'iteration_count'
RELATIVE_ITERATION_COUNT = f'relative_{ITERATION_COUNT}'

# RESULTS---------------------------------------------------------------------------------------------------------------
# Duration
DURATION = 'duration'
RELATIVE_DURATION = f'relative_{DURATION}'
CLUSTER_DURATION = f'clustering_duration'
RELATIVE_CLUSTER_DURATION = f'relative_{CLUSTER_DURATION}'
PREPROCESSING_DURATION = f'preprocessing_duration'
RELATIVE_PREPROCESSING_DURATION = f'relative_{PREPROCESSING_DURATION}'

# External Evaluation Measures
PURITY_WITH_NOISE = f'purity_with_noise'
RELATIVE_PURITY_WITH_NOISE = f'relative_{PURITY_WITH_NOISE}'
PURITY_WITHOUT_NOISE = f'purity_without_noise'
RELATIVE_PURITY_WITHOUT_NOISE = f'relative_{PURITY_WITHOUT_NOISE}'
ARI = 'ari'
RELATIVE_ARI = f'relative_{ARI}'
SILHOUETTE_COEFFICIENT = 'silhouette_coefficient'

# Internal Evaluation Measures
NUMBER_OF_CORE_POINTS = 'number_core_points'
RELATIVE_NUMBER_OF_CORE_POINTS = f'relative_{NUMBER_OF_CORE_POINTS}'
RELATIVE_NOISE = f'relative_{NOISE}'
RELATIVE_NUMBER_OF_FOUND_CLUSTERS = f'relative_{NUMBER_OF_FOUND_CLUSTERS}'

# Other Stuff ----------------------------------------------------------------------------------------------------------
RATIO = 'ratio'
X = 'x'
Y = 'y'
COLOUR_SCALE = 'colour_scale'
PREDICTED = 'predicted'
ABBREVIATION = 'abbreviation'
MEAN_ERROR = f'mean_error'
MEAN_RELATIVE_ERROR = f'mean_relative_error'
MAX_ERROR = f'max_error'
MAX_RELATIVE_ERROR = f'max_relative_error'
AVG_DISTANCE = 'avg_distance'
AVG_4DIST = 'avg_4dist'
