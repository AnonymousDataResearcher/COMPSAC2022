import strings

JKKC = 'JKKC'
AMAX_BMIN = 'AMBM'
EUCLIDEAN_VANILLA = 'EVAN'
MANHATTAN_VANILLA = 'MVAN'
WEIGHTED_JACCARD_VANILLA = 'WJVAN'
ALL_VANILLA_METRICS = [EUCLIDEAN_VANILLA, MANHATTAN_VANILLA, WEIGHTED_JACCARD_VANILLA]
AQBC = 'AQBC'
BITBOOSTED_AQBC = 'BBAQBC'

EUCLIDEAN_BITBOOSTER = 'EBB'
MANHATTAN_BITBOOSTER = 'MBB'
WEIGHTED_JACCARD_MIN_HASH = 'WJMH'


class Metric:

    def __init__(self, s):
        assert self.is_valid_metric_implementation(s)
        self.s = s

    def __str__(self):
        return self.s

    # Metric properties ------------------------------------------------------------------------------------------------
    @property
    def sample_size(self):
        if self.is_wjmh:
            return int(self.s[4:])
        else:
            raise ValueError(f'sample_size not available for {self.s}')

    @property
    def sample_size_or_0(self):
        try:
            return self.sample_size
        except ValueError:
            return 0

    @property
    def n_bits(self):
        if self.is_bitbooster:
            return int(self.s[3:])
        else:
            raise ValueError(f'n_bits not available for {self.s}')

    @property
    def n_bits_or_0(self):
        try:
            return self.n_bits
        except ValueError:
            return 0

    @property
    def base_metric(self):
        if self.s.startswith(EUCLIDEAN_BITBOOSTER) or self.s in [EUCLIDEAN_VANILLA, JKKC, AMAX_BMIN]:
            return strings.EUCLIDEAN
        if self.s.startswith(MANHATTAN_BITBOOSTER) or self.s in [MANHATTAN_VANILLA]:
            return strings.MANHATTAN
        if self.s.startswith(WEIGHTED_JACCARD_MIN_HASH) or self.s == WEIGHTED_JACCARD_VANILLA:
            return strings.WEIGHTED_JACCARD
        if self.s in [AQBC, BITBOOSTED_AQBC]:
            return strings.EUCLIDEAN
        raise NotImplementedError(self.s)

    @property
    def base_metric_capitalized(self):
        return ' '.join([s.capitalize() for s in self.base_metric.split('_')])

    # Static methods ---------------------------------------------------------------------------------------------------
    @staticmethod
    def metric_implementations(metric_code):
        if not isinstance(metric_code, str):
            return sum([Metric.metric_implementations(mc) for mc in metric_code], [])

        if metric_code == strings.EUCLIDEAN:
            return [EUCLIDEAN_VANILLA] + \
                   [f'{EUCLIDEAN_BITBOOSTER}{i}' for i in range(1, 4)] + \
                   [JKKC, AMAX_BMIN, AQBC, BITBOOSTED_AQBC]
        if metric_code == strings.MANHATTAN:
            return [MANHATTAN_VANILLA] + [f'{MANHATTAN_BITBOOSTER}{i}' for i in range(1, 4)]
        if metric_code == strings.WEIGHTED_JACCARD:
            return [WEIGHTED_JACCARD_VANILLA] + [f'{WEIGHTED_JACCARD_MIN_HASH}{i}' for i in range(1, 10)]
        raise NotImplementedError(metric_code)

    @staticmethod
    def vanilla_implementations(metric_code):
        if not isinstance(metric_code, str):
            return [Metric.vanilla_implementations(mc) for mc in metric_code]

        if metric_code == strings.EUCLIDEAN:
            return EUCLIDEAN_VANILLA
        if metric_code == strings.MANHATTAN:
            return MANHATTAN_VANILLA
        if metric_code == strings.WEIGHTED_JACCARD:
            return WEIGHTED_JACCARD_VANILLA
        raise NotImplementedError(metric_code)

    @staticmethod
    def is_valid_metric(s):
        if isinstance(s, str):
            return s in [strings.EUCLIDEAN, strings.MANHATTAN, strings.WEIGHTED_JACCARD]
        else:
            return all([Metric.is_valid_metric(si) for si in s])

    @staticmethod
    def is_valid_metric_implementation(s):
        if not isinstance(s, str):
            return False
        if s.startswith(EUCLIDEAN_BITBOOSTER) or s.startswith(MANHATTAN_BITBOOSTER):
            return s[3:] in ['1', '2', '3']
        if s in [EUCLIDEAN_VANILLA, MANHATTAN_VANILLA, JKKC, AMAX_BMIN, WEIGHTED_JACCARD_VANILLA, AQBC,
                 BITBOOSTED_AQBC]:
            return True
        if s.startswith(WEIGHTED_JACCARD_MIN_HASH):
            return s[4:].isdigit()
        return False

    # Visualization properties -----------------------------------------------------------------------------------------
    @property
    def short_title(self):
        if self.is_bitbooster:
            return str(self)
        elif self.s == BITBOOSTED_AQBC:
            return AQBC.upper()
        elif self.s == AQBC:
            return AQBC.lower()
        return self.title

    @property
    def legend_name(self):
        return self.title. \
            replace(strings.EUCLIDEAN.capitalize(), strings.EUC). \
            replace(strings.MANHATTAN.capitalize(), strings.MAN)

    @property
    def title(self):
        if self.is_vanilla:
            return self.base_metric_capitalized
        if self.is_bitbooster:
            return f'{self.base_metric_capitalized} BB{self.n_bits}'
        if self.is_wjmh:
            return f'WJMinHash{self.sample_size}'
        if self.s == JKKC:
            return 'JKKC'
        if self.s == AMAX_BMIN:
            return r'$\alpha$MAX$\beta$MIN'
        if self.s == AQBC:
            return AQBC.lower()
        if self.s == BITBOOSTED_AQBC:
            return AQBC.upper()
        raise NotImplementedError(self.s)

    @property
    def marker_dict(self):
        d = dict()

        if self.base_metric == strings.EUCLIDEAN:
            d['fillstyle'] = 'full'
        elif self.base_metric == strings.MANHATTAN:
            d['fillstyle'] = 'none'
        elif self.base_metric == strings.WEIGHTED_JACCARD:
            d['fillstyle'] = 'bottom'
        else:
            raise NotImplementedError(self.base_metric)

        if self.is_vanilla:
            d['color'] = 'silver'
            d['marker'] = 'o'
        elif self.is_bitbooster:
            d['color'] = ' byr'[self.n_bits]
            d['marker'] = ' v^>'[self.n_bits]
        elif self.is_wjmh:
            d['color'] = (0, 1, 0, 0.1 * self.sample_size)
            d['marker'] = 'h'
        elif self.s == JKKC:
            d['color'] = 'c'
            d['marker'] = 's'
        elif self.s == AMAX_BMIN:
            d['color'] = 'm'
            d['marker'] = '*'
        elif self.s == AQBC:
            d['color'] = 'g'
            d['marker'] = 'D'
        elif self.s == BITBOOSTED_AQBC:
            d['color'] = 'g'
            d['marker'] = 'd'
        else:
            raise NotImplementedError(self.s)

        return d

    @property
    def short_tex(self):
        if self.is_bitbooster:
            return rf'\texttt{{{str(self)}}}'
        elif self.s == BITBOOSTED_AQBC:
            return rf'\texttt{{{AQBC.upper()}}}'
        elif self.s == AQBC:
            return rf'\texttt{{{AQBC.lower()}}}'
        return self.tex

    @property
    def tex(self):
        if self.s == AMAX_BMIN:
            return r'$\alpha$\texttt{MAX}$\beta$\texttt{MIN}'
        if self.is_vanilla:
            return self.base_metric_capitalized
        if self.s == BITBOOSTED_AQBC:
            return r'\texttt{AQBC}'
        if self.is_bitbooster:
            return rf'{self.base_metric_capitalized} \texttt{{BB}}{self.n_bits}'
        if self.s == JKKC:
            return r'\texttt{JKKC}'
        if self.is_wjmh:
            return fr'\texttt{{WJMH}}{self.sample_size}'
        if self.s == AQBC:
            return r'\texttt{aqbc}'
        raise NotImplementedError(self.s)

    # Type properties --------------------------------------------------------------------------------------------------
    # Philosophical questions: Can we make this a better coding standard by for example replacing is_vanilla by
    # 'return self.s in [...]'? Yes we could; but in this way the addition of a new metric forces the implementation
    # to define each property. Can we refer to 'is_bitbooster' in 'is_vanilla', instead of writing
    # 'self.s.startswith(EUCLIDEAN_BITBOOSTER)...' twice? Yes we could; but that would mean that metric can never have
    # multiple properties; which might not seem like a bad thing; but in theory a combination of BitBooster and WJ(MH)
    # is possible. I feel like this is the best combination between coding standards and future proof-ness.

    # Edit on 22-09-2021: I am happy I did this; as I now have a BitBoosted + AQBC implementation, and the below forced
    # me to consider what that actually means

    @property
    def is_vanilla(self):
        if self.s.startswith(EUCLIDEAN_BITBOOSTER) or self.s.startswith(MANHATTAN_BITBOOSTER):
            return False
        if self.is_wjmh:
            return False
        if self.s in [AQBC, BITBOOSTED_AQBC]:
            return False
        if self.s in [EUCLIDEAN_VANILLA, MANHATTAN_VANILLA, WEIGHTED_JACCARD_VANILLA]:
            return True
        if self.s in [JKKC, AMAX_BMIN]:
            return False
        raise NotImplementedError(self.s)

    @property
    def is_bitbooster(self):
        if self.s.startswith(EUCLIDEAN_BITBOOSTER) or self.s.startswith(MANHATTAN_BITBOOSTER):
            return True
        if self.s.startswith(WEIGHTED_JACCARD_MIN_HASH):
            return False
        if self.s in [EUCLIDEAN_VANILLA, MANHATTAN_VANILLA, WEIGHTED_JACCARD_VANILLA, JKKC, AMAX_BMIN, AQBC]:
            return False
        if self.s == BITBOOSTED_AQBC:
            return False
            # Even though we use the bitbooster implementation for this; this is not the actual bitbooster
        raise NotImplementedError(self.s)

    @property
    def is_aqbc(self):
        if self.s.startswith(EUCLIDEAN_BITBOOSTER) or self.s.startswith(MANHATTAN_BITBOOSTER):
            return False
        if self.s.startswith(WEIGHTED_JACCARD_MIN_HASH):
            return False
        if self.s in [EUCLIDEAN_VANILLA, MANHATTAN_VANILLA, WEIGHTED_JACCARD_VANILLA, JKKC, AMAX_BMIN]:
            return False
        if self.s in [BITBOOSTED_AQBC, AQBC]:
            return True
        raise NotImplementedError(self.s)

    @property
    def is_wjmh(self):
        if self.s.startswith(EUCLIDEAN_BITBOOSTER) or self.s.startswith(MANHATTAN_BITBOOSTER):
            return False
        if self.s.startswith(WEIGHTED_JACCARD_MIN_HASH):
            return True
        if self.s in [EUCLIDEAN_VANILLA, MANHATTAN_VANILLA, WEIGHTED_JACCARD_VANILLA, JKKC, AMAX_BMIN, AQBC,
                      BITBOOSTED_AQBC]:
            return False
        raise NotImplementedError(self.s)
