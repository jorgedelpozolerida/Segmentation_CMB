import logging
from typing import Dict, Optional
from collections import defaultdict
import numpy as np
import clearml
from crbr.evaluation.evaluations import Evaluation, AccumulatedEvaluation
from crbr.evaluation.classification_outcome import ClassificationOutcome
from crbr.utils.evaluation_utils import compute_metrics_from_cm

logger = logging.getLogger(__name__)


class ClassificationEvaluation(Evaluation):
    """
    Basic classification evaluation class for comparing two multi-class binary vectors. Returns a
    dictionary with each class name as key and a ClassificationOutcome object 
    (representing FP/FN/TP/TN outcomes) as value.
    """

    NAME = "classification"

    def __init__(self, **eval_params):
        """
        :param class_label_map: An optional dictionary mapping class indices to class names.
            This will change keys in the metrics dict returned by self.evaluate. E.g., for
            easier readability. Example: {0: "no tumour", 1: "tumour"}.
        """
        super().__init__(**eval_params)

    def evaluate(self, true_array: np.ndarray, predicted_array: np.ndarray) -> dict:
        """
        Evaluate a shape [N] multi-class binary vector against a shape [N] multi-class binary
        predicted vector.

        :param true_array: The true multi-class binary array
        :param predicted_array: The predicted multi-class binary array
        :return: A dictionary of metrics.
        """
        # Check shapes
        if len(true_array) != len(predicted_array):
            raise ValueError(
                "true_array and predicted_array must have the same shape, "
                f"got {true_array.shape} and {predicted_array.shape}"
            )
        # Cast to bool (mostly for input validation purposes)
        true_array = true_array.astype(bool)
        predicted_array = predicted_array.astype(bool)

        # Get a list of class labels
        classes = np.arange(len(true_array))

        metrics_dict = {}
        for class_, true, predicted in zip(classes, true_array, predicted_array):
            outcome = ClassificationOutcome.from_observations(
                observed=true, predicted=predicted
            )
            metrics_dict[class_] = outcome

        return metrics_dict


class AcccumulatedClassificationEvaluation(
    ClassificationEvaluation, AccumulatedEvaluation
):
    """
    Accumulated classification evaluation class. Mirrors the SegmentationEvaluation class but
    each call the self.evaluate updates an internal confusion matrix. Metrics are finally computed
    in self.get_accumulated_evaluation, which calls the self.compute_metrics_from_cm method on the
    accumulated CM. The CM may be reset with self.reset_accumulator.

    See AccumulatedEvaluation for more details.
    """

    NAME = "accumulated_classification"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create multi-label confusion matrices
        self.accumulated_cms = defaultdict(lambda: np.zeros((2, 2), dtype=int))

    def _reset_accumulator(self):
        """
        Resets the internal confusion matrix counts by filling them with zeros.
        """
        for confusion_matrix in self.accumulated_cms.values():
            confusion_matrix.fill(0)

    def compute_metrics_from_binary_cm(self, confusion_matrix: np.ndarray) -> dict:
        """
        Computes the metrics from a binary confusion matrix.

        :param confusion_matrix: A binary confusion matrix of shape [2, 2], where cm[0, 0] is TN,
            cm[0, 1] is FP, cm[1, 0] is FN, and cm[1, 1] is TP (unless otherwise specified in the
            crbr.utils.evaluation_utils.ClassificationOutcome class).
        :return: A dictionary of metrics for a single class, Dict[str, float]
        """
        metrics_df = compute_metrics_from_cm(confusion_matrix, as_dict=False, add_avg_column=False)
        metric_names = list(metrics_df.T.columns)
        metric_values = metrics_df.iloc[:, 1].values
        return dict(zip(metric_names, metric_values))

    def log_accumulated_evaluation(
        self, logger: clearml.Logger, accumulated_metrics: Optional[dict] = None, round: int = 2
    ) -> None:
        """
        Log the accumulated evaluation.

        :param logger: The ClearML logger to log to.
        :param accumulated_metrics: The accumulated metrics to log. If not specified, will call
            self.get_accumulated_evaluation.
        :param round: The number of decimal places to round to.
        """
        accumulated_metrics = (
            accumulated_metrics or self.get_accumulated_evaluation()
        )

        # Report confusion matrices
        keys = [k for k in accumulated_metrics.keys() if k.startswith('confusion_matrix')]
        classes = [k.split('_')[-1] for k in keys]
        classes = self.safe_apply_label_map(classes)
        for key, class_ in zip(keys, classes):
            # Get CM as DF table without applying label map (multilabel binary CM here)
            cm = self.cm_to_table(
                accumulated_metrics[key],
                apply_label_map=False,
                multi_label_binary=True
            ).round(round)
            logger.report_table(
                title="Accumulated Classification Evaluations",
                series=f"Multi-label CM for class: {class_}",
                table_plot=cm,
            )
        # Report metrics table
        logger.report_table(
            title="Accumulated Classification Evaluations",
            series="Metrics",
            table_plot=self.metrics_to_table(accumulated_metrics).round(round),
        )

    def get_accumulated_evaluation(self) -> dict:
        """
        Computes the metrics from the accumulated confusion matrices.

        :return: A dictionary of metrics for all classes, Dict[str, float].
        """
        class_wise_metrics = {}
        for class_label, confusion_matrix in self.accumulated_cms.items():
            # Compute metrics for the current class
            class_metrics = self.compute_metrics_from_binary_cm(confusion_matrix)

            # Add metrics with label {metric_name}_{class_label}
            for metric_name, metric_value in class_metrics.items():
                class_wise_metrics[f"{metric_name}_{class_label}"] = metric_value

            # Add confusion matrix
            class_wise_metrics[f'confusion_matrix_{class_label}'] = confusion_matrix.copy()

        return class_wise_metrics

    def evaluate(self, true_array: np.ndarray, predicted_array: np.ndarray) -> None:
        """
        Evaluate a shape [N] multi-class binary vector against a shape [N] multi-class binary
        predicted vector, but cashes the results in the internally stored multi-label confiusion
        matrices.

        :param true_array: The true multi-class binary array
        :param predicted_array: The predicted multi-class binary array
        :return: None
        """
        evaluation = super().evaluate(true_array=true_array, predicted_array=predicted_array)
        for class_label, outcome in evaluation.items():
            self.accumulated_cms[class_label][outcome.cm_indices] += 1
