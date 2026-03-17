from typing import List

class MyValueError(ValueError):
    pass

class InputTypeError(ValueError):
    def __init__(self):
        self.message = "Invalid input type, Dict[str, pd.DataFrame] expected by PlanogramsClustering."
        super().__init__(self.message)


class InputStructureError(ValueError):
    def __init__(self, planogram_id: str, columns: List[str]):
        self.message = "Invalid data structure for planogram_id {planogram_id}, " \
                       "the relevant value (dataframe) must have columns {columns}.".format(planogram_id=str(planogram_id),
                                                                                            columns=str(columns))
        super().__init__(self.message)


class NotFittedError(Exception):
    def __init__(self):
        self.message = "This PlanogramsClustering instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
        super().__init__(self.message)


class PlanogramIDError(Exception):
    def __init__(self, planogram_id: str):
        self.message = "No nested clustering assignment available for planogram_id {planogram_id}, " \
                       "it does not belong to the training set.".format(planogram_id=str(planogram_id))
        super().__init__(self.message)


class FitFailedError(Exception):
    def __init__(self, planogram_id_x: str, planogram_id_y: str):
        self.message = "PlanogramsClustering fit failed, incoherent values found comparing" \
                       "planograms {planogram_id_x} and {planogram_id_y}.".format(planogram_id_x=str(planogram_id_x),
                                                                                  planogram_id_y=str(planogram_id_y))
        super().__init__(self.message)

