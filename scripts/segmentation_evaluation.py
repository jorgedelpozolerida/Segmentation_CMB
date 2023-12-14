import logging
from typing import List, Union, Optional
import numpy as np
import clearml
from sklearn.metrics import confusion_matrix as compute_confusion_matrix
from crbr.evaluation.evaluations import Evaluation, AccumulatedEvaluation
from crbr.utils.evaluation_utils import compute_metrics_from_cm

logger = logging.getLogger(__name__)


class SegmentationEvaluation(Evaluation):
    """
    Basic segmentation evaluation class. Returns a dictionary with the following metrics:
      - f1: F1 scores for each class.
      - precision: Precision scores for each class.
      - recall: Recall scores for each class.
      - kappa: Cohen's kappa score.
      - confusion_matrix: The confusion matrix.
    """

    NAME = "segmentation"

    def __init__(
        self,
        zero_division: Union[float, np.nan] = np.nan,
        ignore_classes_in_avg: List[int] = (0,),
        **kwargs,
    ):
        """
        :param zero_division: The value to use when a class has no true positives nor false
            positives (for precision) or no true positives and no false negatives (for recall).
            Should be one of 1.0, 0.0 or np.nan. Default is np.nan.
        :param ignore_classes_in_avg: List of classes to ignore when computing average metrics.
        """
        super().__init__(
            zero_division=zero_division,
            ignore_classes_in_avg=ignore_classes_in_avg,
            **kwargs
        )

    def compute_confusion_matrix(
        self, true_array: np.ndarray, predicted_array: np.ndarray
    ) -> np.ndarray:
        """
        Compute a confusion matrix from a true segmentation array and a predicted segmentation array

        :param true_array: The true segmentation array.
        :param predicted_array: The predicted segmentation array.
        :return: A confusion matrix.
        """
        return compute_confusion_matrix(
            true_array.ravel(),
            predicted_array.ravel(),
            labels=list(range(self.n_classes)),
        )

    def evaluate(self, true_array: np.ndarray, predicted_array: np.ndarray) -> dict:
        """
        Evaluate a true segmentation array and a predicted segmentation array.

        :param true_array: The true segmentation array.
        :param predicted_array: The predicted segmentation array.
        :return: A dictionary of metrics.
        """
        # Compute the confusion matrix
        confusion_matrix = self.compute_confusion_matrix(true_array, predicted_array)

        # Return a DF of segmentation metrics computed from the conf matrix
        return compute_metrics_from_cm(confusion_matrix, as_dict=True)


class AccumulatedSegmentationEvaluation(SegmentationEvaluation, AccumulatedEvaluation):
    """
    Accumulated segmentation evaluation class. Mirrors the SegmentationEvaluation class but
    each call the self.evaluate updates an internal confusion matrix. Metrics are finally computed
    in self.get_accumulated_evaluation, which calls the self.compute_metrics_from_cm method on the
    accumulated CM. The CM may be reset with self.reset_accumulator.

    See AccumulatedEvaluation for more details.
    """

    NAME = "accumulated_segmentation"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulated_cm = np.zeros((self.n_classes,) * 2, dtype=int)

    def _reset_accumulator(self) -> None:
        """
        Resets the internal confusion matrix counts by filling them with zeros.
        """
        self.accumulated_cm.fill(0)

    def log_accumulated_evaluation(
        self, logger: clearml.Logger, accumulated_metrics: Optional[dict] = None, round: int = 2
    ) -> None:
        """
        Log the accumulated evaluation.

        :param logger: The ClearML logger to log to.
        :param accumulated_metrics: The accumulated metrics to log. If not specified, will call
            self.get_accumulated_evaluation.
        :param round: The number of decimals to round the metrics to.
        """
        accumulated_metrics = (
            accumulated_metrics or self.get_accumulated_evaluation()
        )
        # Report summed confusion matrix
        logger.report_table(
            title='Accumulated Segmentation Evaluations',
            series="Confusion Matrix",
            table_plot=self.cm_to_table(accumulated_metrics['confusion_matrix']).round(round),
        )
        # Report metrics table
        logger.report_table(
            title='Accumulated Segmentation Evaluations',
            series="Metrics",
            table_plot=self.metrics_to_table(accumulated_metrics).round(round),
        )

    def get_accumulated_evaluation(self) -> dict:
        """
        Computes the metrics from the accumulated confusion matrix.

        :return: A dictionary of metrics.
        """
        return compute_metrics_from_cm(
            self.accumulated_cm, as_dict=True, include_cm=True
        )

    def evaluate(self, true_array: np.ndarray, predicted_array: np.ndarray) -> None:
        """
        Updates the internal CM with the given true and predicted segmentation arrays.

        :param true_array: The true segmentation array.
        :param predicted_array: The predicted segmentation array.
        :return: None
        """
        # Compute the confusion matrix
        self.accumulated_cm += self.compute_confusion_matrix(
            true_array, predicted_array
        )
