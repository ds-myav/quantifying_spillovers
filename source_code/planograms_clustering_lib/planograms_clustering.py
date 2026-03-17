import numpy as np
import pandas as pd
from typing import Dict
from itertools import combinations
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

from source_code.planograms_clustering_lib.planograms_clustering_exceptions import *


class PlanogramsClustering:
    """  
    This class implements the method proposed in [1].
    It detects 2-level nested clusters in planograms data, with the following hierarchical structure:

        ├── shape_segment_0
        │   ├── cluster_label_0_0
        │   ├── cluster_label_0_1
        │   ├── ...
        │   └── cluster_label_0_N0
        │
        ├── shape_segment_1
        │   ├── cluster_label_1_0
        │   ├── cluster_label_1_1
        │   ├── ...
        │   └── cluster_label_1_N1
        │
        ├── ...
        │
        └── shape_segment_m
            ├── cluster_label_m_0
            ├── cluster_label_m_1
            ├──    ...
            └── cluster_label_m_Nm          
    
    References
    ----------
    [1] F. Silverio et al., A customer behavior-driven clustering method in the planogram design domain, 
    Appl. Soft Comput. 172 (2025) 112836. DOI: https://doi.org/10.1016/j.asoc.2025.112836
    """

    class InputStructure:
        """
        InputStructure.PlanogramFeature, the columns of InputStructure.PlanogramFeature dataframe:
        ['art_code', 'h_facings', 'v_facings', 'unit_width', 'unit_height', 'x_start', 'y_start']

        NOTE:
            - it is assumed that the 'art_code', 'x_start', 'y_start' columns are 'key columns'
            - it is assumed that the 'unit_width', 'unit_height', 'x_start' and 'y_start' values are given in centimeters
        """

        class PlanogramFeature:
            ART_CODE: str = 'art_code'
            H_FACINGS: str = 'h_facings'
            V_FACINGS: str = 'v_facings'
            UNIT_WIDTH: str = 'unit_width'
            UNIT_HEIGTH: str = 'unit_height'
            X_START: str = 'x_start'
            Y_START: str = 'y_start'

            @classmethod
            def get_labels(cls):
                return [value for name, value in vars(cls).items() if (name.isupper())]

    def __init__(self,
                 shape_threshold: float = 90,
                 distance_threshold: float = 0.33,
                 w_h: tuple = (0.25, 0.01, 0.05),  # (omega_x, eta_x, theta_x)
                 w_v: tuple = (1, 0.025, 0.05)):  # (omega_y, eta_y, theta_y)
        """
        Planograms Clustering
        :param shape_threshold: float in [0,inf) (hint: 90 if cm as measurement unit is used)
        :param distance_threshold: float in [0,1], default=0.33
        :param w_h: tuple(float,float,float), default=(0.25, 0.01, 0.05), these are the following params: omega_x, eta_x, theta_x
        :param w_v: tuple(float,float,float), default=(1, 0.025, 0.05), these are the following params: omega_y, eta_y, theta_y

        Remark:
            - we set t_{Y, 1}_{p} = 110 and $t_{Y, 2}_{p} = 150 (note that this defines the eye-level of shelves);
            - while t_{X, 1}_{p} and t_{X, 2}_{p} are assigned by the following heuristic:

            \begin{equation*}
                t_{X, k}^{p} =  \left\lbrace
                \begin{array}{ll}
                x_{k}^{p} \, ,  & \text{if} \; x_{2}^{p} - x_{1}^{p} < w \\
                \dfrac{k}{3} (x_{2}^{p} - x_{1}^{p}) \, ,   & \text{if} \; w \leq x_{2}^{p} - x_{1}^{p} \leq 3w \\
                x_{k}^{p} + (-1)^{k-1}w \, ,   & \text{if} \; x_{2}^{p} - x_{1}^{p} > 3w
                \end{array}
                \right. , \quad w=133 \, , \quad k=1,2 \, , \quad p \in P \, ,
            \end{equation*}
            where
            \begin{equation*}
                x_{1}^{p} = \min_{s \in S^p} \widetilde{c}_{X} (F_{s}^{p}) \quad \text{and} \quad x_{2}^{p} = \max_{s \in S^p} \widetilde{c}_{X} (F_{s}^{p}) \, ,
            \end{equation*}

            and $w=133$ is nothing but the width of any modular component of the shelves in our dataset.
        """
        if shape_threshold < 0:
            raise MyValueError("shape_threshold must be a value in the range [0,inf)")

        self.shape_threshold = shape_threshold

        if distance_threshold < 0 or distance_threshold > 1:
            raise MyValueError("distance_threshold must be a value in the range [0,1]")

        self.distance_threshold = distance_threshold

        # w_h validation
        if len(w_h) != 3:
            raise MyValueError("w_h must be tuple with 3 float values")
        if not (0 < w_h[1] < w_h[2] < w_h[0] <= 1):
            raise MyValueError(f"w_h must verify 0 < w_h[1] < w_h[2] < w_h[0] <= 1")

        # w_v validation
        if len(w_v) != 3:
            raise MyValueError("w_v must be tuple with 3 float values")
        if not (0 < w_v[1] < w_v[2] < w_v[0] <= 1):
            raise MyValueError(f"w_h must verify 0 < w_v[1] < w_v[2] < w_v[0] <= 1")

        # w_h-w_v relationship validation
        if not (w_h[0] <= w_v[0]):
            raise MyValueError(f"must verify w_h[0] <= w_v[0]")

        self.omega_x, self.eta_x, self.theta_x = w_h[0], w_h[1], w_h[2]
        self.omega_y, self.eta_y, self.theta_y = w_v[0], w_v[1], w_v[2]

        self._shape_segments = None
        self._relevant_planogram_ids = None
        self._distances = None
        self._labels = None
        self._model = None

    def fit(self, X: Dict[str, pd.DataFrame]):
        """
        Training instances to nested clustering.
        :param X: dictionary having the following structure:
            - key: string, the planogram_id
            - value: dataframe having format as defined in InputStructure.PlanogramFeature
        """
        if len(X) <= 1:
            raise MyValueError("Found dict with {n_keys} key(s) while a minimum of 2 is required by PlanogramsClustering.".format(n_keys=len(X)))

        self._relevant_planogram_ids = frozenset(X.keys())

        _X = pd.DataFrame()
        for _planogram_id, _planogram_df in X.items():

            if type(_planogram_id) != str or type(_planogram_df) != pd.DataFrame:
                raise InputTypeError()
            if set(_planogram_df.columns) != set(self.InputStructure.PlanogramFeature.get_labels()):
                raise InputStructureError(planogram_id=_planogram_id, columns=self.InputStructure.PlanogramFeature.get_labels())
            
            _planogram_df = _planogram_df.groupby(by=[self.InputStructure.PlanogramFeature.ART_CODE], as_index=False).apply(self._mark_points).reset_index(drop=True)
            _planogram_df = self._x_centroid_labeling(_planogram_df)
            _planogram_df.insert(column='planogram_id', loc=0, value=[_planogram_id] * len(_planogram_df))
            _X = pd.concat([_X, _planogram_df], ignore_index=True)

        
        _X = _X.sort_values(by=['planogram_id', self.InputStructure.PlanogramFeature.ART_CODE], ignore_index=True).reset_index(drop=True)

        X = _X.copy()  # overwrite input X

        X = X.sort_values(by=['planogram_id', self.InputStructure.PlanogramFeature.ART_CODE]).reset_index(drop=True)

        _planograms_peaks_df = self._planograms_segmentation(X)
        X = X.merge(right=_planograms_peaks_df, on='planogram_id', how='left')

        self._distances = pd.DataFrame(columns=['shape_segment',
                                                'relevant_planograms',
                                                'distances'])

        self._shape_segments = frozenset(X['shape_segment'].values)

        for _shape_segment in self._shape_segments:

            _planogram_segment_df = X.loc[X['shape_segment'] == _shape_segment]
            _relevant_planograms = frozenset(_planogram_segment_df['planogram_id'].astype(str).unique())
            _n_samples = len(_relevant_planograms)

            _distances = pd.DataFrame(index=_relevant_planograms, columns=_relevant_planograms, data=float(0))

            #  compute the symmetric matrix _distances
            for planogram_id_x, planogram_id_y in combinations(_relevant_planograms, 2):

                _planogram_x = \
                    _planogram_segment_df.loc[X['planogram_id'] == planogram_id_x].sort_values(by=[self.InputStructure.PlanogramFeature.ART_CODE, 'x_centroid', 'y_centroid']
                                                                                               ).set_index(self.InputStructure.PlanogramFeature.ART_CODE)[['x_centroid', 'y_centroid', 'x_centroid_label', 'y_centroid_label', 'n_facings']]
                _planogram_y = _planogram_segment_df.loc[X['planogram_id'] == planogram_id_y].sort_values(by=[self.InputStructure.PlanogramFeature.ART_CODE, 'x_centroid', 'y_centroid']
                                                                                               ).set_index(self.InputStructure.PlanogramFeature.ART_CODE)[['x_centroid', 'y_centroid', 'x_centroid_label', 'y_centroid_label', 'n_facings']]

                if len(set(_planogram_x.index).intersection(set(_planogram_y.index))) == 0:  # empty intersection on art_code between _planogram_x and _planogram_y
                    _distances.at[planogram_id_x, planogram_id_y] = 1
                    _distances.at[planogram_id_y, planogram_id_x] = 1
                    continue

                _n_art_codes_intersection = len(set(_planogram_x.index).intersection(set(_planogram_y.index)))
                _n_art_codes_union = len(set(_planogram_x.index).union(set(_planogram_y.index)))
                _art_codes_iou = _n_art_codes_intersection / _n_art_codes_union

                ######################
                # Distance Evaluation                
                ######################

                _training_df = pd.merge(_planogram_x, _planogram_y, suffixes=("_1", "_2"), left_index=True, right_index=True, how='outer')
                _n_art_codes_diff = len(
                    _training_df.loc[(_training_df['x_centroid_1'] != _training_df['x_centroid_2']) | (_training_df['y_centroid_1'] != _training_df['y_centroid_2'])])

                if _n_art_codes_diff > 0:

                    _centroids = np.concatenate((np.array(_training_df[['x_centroid_1', 'y_centroid_1']].dropna().to_numpy(float)),
                                                 np.array(_training_df[['x_centroid_2', 'y_centroid_2']].dropna().to_numpy(float))))
                    # Variance Analysis
                    _centroids_var = np.var(_centroids, axis=0)
                    if _centroids_var[0] == 0 and _centroids_var[1] != 0:
                        k_x = 0
                        k_y = 1
                    elif _centroids_var[0] != 0 and _centroids_var[1] == 0:
                        k_x = 1
                        k_y = 0
                    elif _centroids_var[0] == 0 and _centroids_var[1] == 0:
                        k_x = 0
                        k_y = 0
                    else:
                        k_x = 1
                        k_y = 1

                    _training_df['omega_x'] = k_x
                    _training_df['omega_y'] = k_y

                    ###########################################
                    # x (horizontal) swap condition evaluation
                    ###########################################

                    # Different labels, with at least one central point (x_centroid_label = 1)
                    _high_critical_x_swap_condition = (_training_df['x_centroid_1'] != _training_df['x_centroid_2']) & \
                                                      ((_training_df['x_centroid_1'] != 0) & (_training_df['x_centroid_2'] != 0)) & \
                                                      (_training_df['x_centroid_label_1'] != _training_df['x_centroid_label_2']) & \
                                                      ((_training_df['x_centroid_label_1'] == 1) | (_training_df['x_centroid_label_2'] == 1))
                    _training_df.loc[_high_critical_x_swap_condition, 'omega_x'] = k_x * self.omega_x

                    # Different labels, with no central point (x_centroid_label != 1)
                    _almost_critical_x_swap_condition = (_training_df['x_centroid_1'] != _training_df['x_centroid_2']) & \
                                                        ((_training_df['x_centroid_1'] != 0) & (_training_df['x_centroid_2'] != 0)) & \
                                                        (_training_df['x_centroid_label_1'] != _training_df['x_centroid_label_2']) & \
                                                        ((_training_df['x_centroid_label_1'] != 1) & (_training_df['x_centroid_label_2'] != 1))
                    _training_df.loc[_almost_critical_x_swap_condition, 'omega_x'] = k_x * self.eta_x

                    # Same labels
                    _low_critical_x_swap_condition = (_training_df['x_centroid_1'] != _training_df['x_centroid_2']) & \
                                                     ((_training_df['x_centroid_1'] != 0) & (_training_df['x_centroid_2'] != 0)) & \
                                                     (_training_df['x_centroid_label_1'] == _training_df['x_centroid_label_2'])
                    _training_df.loc[_low_critical_x_swap_condition, 'omega_x'] = k_x * self.theta_x

                    # Same positions
                    _no_swap_condition = (_training_df['x_centroid_1'] == _training_df['x_centroid_2'])
                    _training_df.loc[_no_swap_condition, 'omega_x'] = 0

                    # Exists in just one planogram
                    _only_one_planogram_condition = (_training_df['x_centroid_1'].isna() | _training_df['x_centroid_2'].isna())
                    _training_df.loc[_only_one_planogram_condition, 'omega_x'] = 0

                    #########################################
                    # y (vertical) swap condition evaluation
                    #########################################

                    # Different labels, with at least one central point (y_centroid_label = 1)
                    _high_critical_y_swap_condition = (_training_df['y_centroid_1'] != _training_df['y_centroid_2']) & \
                                                      ((_training_df['y_centroid_1'] != 0) & (_training_df['y_centroid_2'] != 0)) & \
                                                      (_training_df['y_centroid_label_1'] != _training_df['y_centroid_label_2']) & \
                                                      ((_training_df['y_centroid_label_1'] == 1) | (_training_df['y_centroid_label_2'] == 1))
                    _training_df.loc[_high_critical_y_swap_condition, 'omega_y'] = k_y * self.omega_y

                    # Different labels, with no central point (y_centroid_label != 1)
                    _almost_critical_y_swap_condition = (_training_df['y_centroid_1'] != _training_df['y_centroid_2']) & \
                                                        ((_training_df['y_centroid_1'] != 0) & (_training_df['y_centroid_2'] != 0)) & \
                                                        (_training_df['y_centroid_label_1'] != _training_df['y_centroid_label_2']) & \
                                                        ((_training_df['y_centroid_label_1'] != 1) & (_training_df['y_centroid_label_2'] != 1))
                    _training_df.loc[_almost_critical_y_swap_condition, 'omega_y'] = k_y * self.eta_y

                    # Same labels
                    _low_critical_y_swap_condition = (_training_df['y_centroid_1'] != _training_df['y_centroid_2']) & \
                                                     ((_training_df['y_centroid_1'] != 0) & (_training_df['y_centroid_2'] != 0)) & \
                                                     (_training_df['y_centroid_label_1'] == _training_df['y_centroid_label_2'])
                    _training_df.loc[_low_critical_y_swap_condition, 'omega_y'] = k_y * self.theta_y

                    # Same positions
                    _no_swap_condition = (_training_df['y_centroid_1'] == _training_df['y_centroid_2'])
                    _training_df.loc[_no_swap_condition, 'omega_y'] = 0

                    # Exists in just one planogram
                    _only_one_planogram_condition = (_training_df['y_centroid_1'].isna() | _training_df['y_centroid_2'].isna())
                    _training_df.loc[_only_one_planogram_condition, 'omega_y'] = 0
                    
                    #########################################
                    
                    _weights = _training_df[['omega_x', 'omega_y']].to_numpy(float).flatten()

                    _x_centroid_max = _training_df[['x_centroid_1', 'x_centroid_2']].max().max()
                    _min_value = _training_df[['y_centroid_1', 'y_centroid_2']].min().min()
                    _max_value = _training_df[['y_centroid_1', 'y_centroid_2']].max().max()

                    if _min_value == _max_value:  # assign min_value if centroids have same height
                        _min_value = 0.1 * _max_value

                    _p1 = _training_df[['x_centroid_1', 'y_centroid_1']].to_numpy(float, na_value=np.nan)
                    if _max_value <= _x_centroid_max:  # compress centroid data if height <= width
                        scaler = MinMaxScaler(feature_range=(_min_value, _max_value))
                        _p1 = scaler.fit(_p1).transform(_p1)
                    _p1 = np.nan_to_num(_p1, nan=float(0)).flatten()

                    _p2 = _training_df[['x_centroid_2', 'y_centroid_2']].to_numpy(float, na_value=np.nan)
                    if _max_value <= _x_centroid_max:  # compress centroid data if height <= width
                        scaler = MinMaxScaler(feature_range=(_min_value, _max_value))
                        _p2 = scaler.fit(_p2).transform(_p2)
                    _p2 = np.nan_to_num(_p2, nan=float(0)).flatten()

                    _max_value = _max_value if _max_value <= _x_centroid_max else _x_centroid_max

                    _weighted_p1 = np.multiply(_weights, _p1)
                    _weighted_p2 = np.multiply(_weights, _p2)
                    _j = _art_codes_iou
                    _e = distance.euclidean(u=_weighted_p1, v=_weighted_p2) / (_max_value * np.sqrt(2 * _n_art_codes_union * (self.omega_x + self.omega_y)))
                    _d = 1 - (_j / (1 + _e))
                    _distances.at[planogram_id_x, planogram_id_y] = _d
                    _distances.at[planogram_id_y, planogram_id_x] = _d

                    
            _row = {
                'shape_segment': _shape_segment,
                'relevant_planograms': _relevant_planograms,
                'distances': _distances.to_json(orient='table')
            }

            self._distances = pd.concat([self._distances, pd.DataFrame([_row])], ignore_index=True)

            self._labels = pd.DataFrame(columns=['planogram_id',
                                                 'shape_segment',
                                                 'cluster_label'])

            for _, _row in self._distances.iterrows():
                relevant_planograms = _row['relevant_planograms']
                _curr_shape_segment = _row['shape_segment']

                distances = pd.read_json(_row['distances'], orient='table')

                if len(relevant_planograms) > 1:
                    try:
                        ac = AgglomerativeClustering(n_clusters=None,
                                                 distance_threshold=self.distance_threshold,
                                                 metric='precomputed',
                                                 linkage='average')
                    except:
                        ac = AgglomerativeClustering(n_clusters=None,
                                                 distance_threshold=self.distance_threshold,
                                                 affinity='precomputed',
                                                 linkage='average')
                    ac.fit_predict(distances.values.tolist())
                    for planogram_id, cluster_label in zip(distances, ac.labels_):
                        self._labels = pd.concat([self._labels, pd.DataFrame([{'planogram_id': planogram_id, 'shape_segment': _curr_shape_segment, 'cluster_label': cluster_label}])], ignore_index=True)

                else:
                    self._labels = pd.concat([ self._labels, pd.DataFrame([{ 'planogram_id': list(relevant_planograms)[0], 'shape_segment': _curr_shape_segment, 'cluster_label': 0 }]) ], ignore_index=True)

            self._labels = self._labels.sort_values(by=['shape_segment', 'cluster_label', 'planogram_id']).reset_index(drop=True)

            # Output Model definition
            self._model = dict()
            for _key in self._labels.groupby('shape_segment')['planogram_id'].apply(frozenset):
                self._model[_key] = tuple(self._labels.loc[self._labels['planogram_id'].isin(_key)].groupby('cluster_label')['planogram_id'].apply(frozenset).values)


    def fit_predict(self, X: Dict[str, pd.DataFrame]):
        """
        Fit and return the result of each sample's ('planogram_id') nested clustering assignment.
        :param X: dictionary having the following structure:
            - key: string, the planogram_id
            - value: dataframe having format as defined in InputStructure.PlanogramFeature
        :return: cluster labels as dataframe with following columns: ['planogram_id', 'shape_segment', 'cluster_label']
        """
        self.fit(X)
        return self._labels

    @property
    def model(self):
        """
        Dictionary that defines the nested clustering model having the following structure:
            - key: a frozenset of planogram_id with the same shape_segment
            - value: a tuple of clusters, any cluster being a frozenset of planogram_id with the same cluster_label

        Example:
        {
            frozenset({'planogram_id1', 'planogram_id2','planogram_id5'}): (frozenset({'planogram_id2'}), frozenset({'planogram_id1', 'planogram_id5'})),
            frozenset({'planogram_id3', 'planogram_id4'}): (frozenset({'planogram_id3', 'planogram_id4'})),
            ...
        }
        """
        if self._model is None:
            raise NotFittedError()
        return self._model

    @property
    def labels(self):
        """
        Dataframe with following columns: ['planogram_id', 'shape_segment', 'cluster_label']
        """
        if self._labels is None:
            raise NotFittedError()
        return self._labels

    @property
    def computed_distances(self):
        """
        Dataframe of distances with following columns:
        ['shape_segment', 'relevant_planograms', 'distances']
        The distance matrices are in 'distances' columns as json (pandas dataframe, orient='table)
        """
        if self._distances is None:
            raise NotFittedError()

        return self._distances

    @property
    def relevant_planogram_ids(self):
        """
        Frozenset of the relevant planogram ids
        """
        if self._relevant_planogram_ids is None:
            raise NotFittedError()

        return self._relevant_planogram_ids

    @property
    def shape_segments(self):
        """
        Frozenset of the computed shape segments
        """
        if self._shape_segments is None:
            raise NotFittedError()

        return self._shape_segments

    def shape_segment_items_mapping(self):
        """
        :return: a dictionary that defines the segmentation of training set:
            - key: the shape_segment as integer
            - value: frozenset of relevant planogram_id

        Example:
        {
            0: frozenset({'planogram_id2'}, {'planogram_id1, 'planogram_id5'}),
            1: frozenset({'planogram_id3, 'planogram_id4'}),
            ...
        }
        """
        if self._labels is None:
            raise NotFittedError()

        return self._labels.groupby('shape_segment')['planogram_id'].apply(frozenset).to_dict()

    def distance(self, shape_segment):
        """
        :param shape_segment: the identifier of shape_segment
        :return: the dataframe of distance evaluated between planograms in the given shape_segment
        """
        if self._distances is None:
            raise NotFittedError()

        return pd.read_json(self._distances['distances'].values[shape_segment], orient='table')

    def retrieve_label(self, planogram_id: str = None):
        """
        :param planogram_id: str
        :return: dataframe with following columns: ['planogram_id', 'shape_segment', 'cluster_label']
        """
        if self._labels is None:
            raise NotFittedError()

        _labels = self._labels.loc[self._labels['planogram_id'] == planogram_id].reset_index(drop=True) if planogram_id is not None else self.labels
        if _labels.empty:
            raise PlanogramIDError(planogram_id=planogram_id)

        return _labels

    def retrieve_similar_planograms(self, planogram_id: str):
        """
        :param planogram_id: str
        :return: a set of items with same nested clustering labels
        """
        if self._labels is None:
            raise NotFittedError()

        _planogram_label = self._labels.loc[self._labels['planogram_id'] == planogram_id]
        if _planogram_label.empty:
            raise PlanogramIDError(planogram_id=planogram_id)

        _shape_segment, _cluster_label = _planogram_label[['shape_segment', 'cluster_label']].values[0]

        _similar_items = set(self._labels.loc[(self._labels['shape_segment'] == _shape_segment) & (self._labels['cluster_label'] == _cluster_label), 'planogram_id'].values)

        return _similar_items

    ##################
    # Private methods
    ##################
    def _mark_points(self, _df):
        art_code = int(_df[self.InputStructure.PlanogramFeature.ART_CODE].values[0])
        h_facings = _df[self.InputStructure.PlanogramFeature.H_FACINGS].values
        v_facings = _df[self.InputStructure.PlanogramFeature.V_FACINGS].values
        unit_width = _df[self.InputStructure.PlanogramFeature.UNIT_WIDTH].values
        unit_height = _df[self.InputStructure.PlanogramFeature.UNIT_HEIGTH].values
        x_start = _df[self.InputStructure.PlanogramFeature.X_START].values
        y_start = _df[self.InputStructure.PlanogramFeature.Y_START].values

        n_facings = 0
        _points = []

        for _h_facings, _v_facings, _unit_width, _unit_height, _x_start, _y_start in zip(h_facings, v_facings, unit_width, unit_height, x_start, y_start):
            _h_facings = int(_h_facings)
            _v_facings = int(_v_facings)
            _unit_width = float(_unit_width)
            _unit_height = float(_unit_height)
            _x_start = float(_x_start)
            _y_start = float(_y_start)
            n_facings += _h_facings * _v_facings
            for __v_facings in range(0, _v_facings):
                for __h_facings in range(0, _h_facings):
                    _x1 = _x_start + __h_facings * _unit_width
                    _y1 = _y_start + __v_facings * _unit_height
                    _x2 = _x1 + _unit_width
                    _y2 = _y1 + _unit_height

                    _facing_centroid = np.mean([(_x1, _y1), (_x2, _y2)], axis=0)
                    _points.append(_facing_centroid)

        if _points:
            x_centroid, y_centroid = np.mean(_points, axis=0)
            x_peak, y_peak = np.max(_points, axis=0)

            # vertical position classification function
            def __y_centroid_labeling(_y_centroid):
                if y_centroid < 110:
                    return 0
                elif 110 <= y_centroid <= 150:
                    return 1
                else:
                    return 2

            mark_points = pd.DataFrame(data={self.InputStructure.PlanogramFeature.ART_CODE: [art_code],
                                             'n_facings': [n_facings],
                                             'x_centroid': [x_centroid],
                                             'y_centroid': [y_centroid],
                                             'y_centroid_label': [__y_centroid_labeling(_y_centroid=y_centroid)],
                                             'x_peak': [x_peak],
                                             'y_peak': [y_peak]})

            return mark_points

    @staticmethod
    def _x_centroid_labeling(_df):
        _df.insert(column='x_centroid_label',
                   loc=len(_df.columns),
                   value=0)

        _x_centroid_min = _df['x_centroid'].min()
        _x_centroid_max = _df['x_centroid'].max()

        # horizontal position classification function
        def __x_centroid_labeling(_x_centroid, _x_centroid_min, _x_centroid_max):

            if _x_centroid_max - _x_centroid_min < 133:
                return 1

            offset = (_x_centroid_max - _x_centroid_min) / 3 if 133 <= (_x_centroid_max - _x_centroid_min) <= 133 * 3 else 133

            if _x_centroid < _x_centroid_min + offset:
                return 0
            elif _x_centroid > _x_centroid_max - offset:
                return 2
            else:
                return 1

        _df['x_centroid_label'] = _df['x_centroid'].apply(__x_centroid_labeling, args=(_x_centroid_min, _x_centroid_max))

        return _df

    def _planograms_segmentation(self,
                                 X: pd.DataFrame):
        _planograms_peaks_df = X.groupby(by='planogram_id', as_index=False)[['x_peak', 'y_peak']] \
            .apply(max).rename(columns={'x_peak': 'x_right_up_peak',
                                        'y_peak': 'y_right_up_peak'})
        try:
            _planograms_peaks_df.insert(column='shape_segment',
                                        loc=len(_planograms_peaks_df.columns),
                                        value=AgglomerativeClustering(n_clusters=None,
                                                                      distance_threshold=self.shape_threshold,
                                                                      metric='euclidean',
                                                                      linkage='average').fit_predict(X=_planograms_peaks_df[['x_right_up_peak',
                                                                                                                              'y_right_up_peak']]))
        except:
            _planograms_peaks_df.insert(column='shape_segment',
                                    loc=len(_planograms_peaks_df.columns),
                                    value=AgglomerativeClustering(n_clusters=None,
                                                                  distance_threshold=self.shape_threshold,
                                                                  affinity='euclidean',
                                                                  linkage='average').fit_predict(X=_planograms_peaks_df[['x_right_up_peak',
                                                                                                                          'y_right_up_peak']]))

        return _planograms_peaks_df[['planogram_id',
                                     'shape_segment']]