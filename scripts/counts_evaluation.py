import logging
from typing import List, Union, Dict, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
import clearml
from scipy.ndimage import label, generate_binary_structure
from crbr.evaluation.evaluations import Evaluation, AccumulatedEvaluation

logger = logging.getLogger(__name__)


class CountsEvaluation(Evaluation):
    """
    Basic foreground connected components evaluation class. Returns a dictionary with the following 
    metrics:
        - 'true_counts': a list of true foreground connected components counts
        - 'predicted_counts': a list of predicted foreground connected components counts
        - 'predicted_overlapping_counts': a list of predicted foreground connected components counts
            that overlap with at least 'min_overlap_voxels' voxels to a true foreground connected 
            component of the same class
    """

    NAME = "counts"

    def __init__(
        self,
        n_classes: int,
        label_structure: np.ndarray = generate_binary_structure(3, 3),
        min_overlap_voxels: int = 1,
        **kwargs,
    ):
        """
        :param n_classes: The number of classes where class 0 is the background.
        :param label_structure: The structure used for the connected components labeling. Defaults
            to a 3x3x3 cube.
        :param min_overlap_voxels: The minimum number of overlapping voxels between a predicted
            connected component and a true connected component of the same class to count the
            predicted connected component as overlapping. This only affects the 
            'predicted_overlapping_counts' metric, not the 'predicted_raw_counts' metric.
        :param kwargs: Additional keyword arguments that are not used.
        """
        super().__init__(
            n_classes=n_classes,
            label_structure=label_structure,
            min_overlap_voxels=min_overlap_voxels,
        )

    def evaluate(self, true_array: np.ndarray, predicted_array: np.ndarray) -> dict:
        """
        Evaluate the given true and predicted segmentation arrays by counting the number of
        connected components in the foreground of each image and returning a dictionary with the
        counts as a list of length (n_classes - 1).
        """
        results = defaultdict(list)
        for class_id in range(1, self.n_classes):
            true_array_class = true_array == class_id
            predicted_array_class = predicted_array == class_id
            true_array_class, num_true_components = label(
                true_array_class, structure=self.eval_params['label_structure']
            )
            predicted_array_class, num_predicted_components = label(
                predicted_array_class, structure=self.eval_params['label_structure']
            )
            results['true_counts'].append(num_true_components)
            results['predicted_counts'].append(num_predicted_components)

            # Calculate the number of predicted connected components that overlap with at least
            # 'min_overlap_voxels' voxels with a true connected component of the same class.
            predicted_overlapping_counts = 0
            for predicted_cc in range(1, num_predicted_components + 1):
                # Create a mask for the current predicted connected component.
                predicted_cc_mask = predicted_array_class == predicted_cc

                # Check how large the overlap is between the current predicted connected component
                # and the true_array_class
                overlap = np.logical_and(predicted_cc_mask, true_array_class)
                overlap_size = np.sum(overlap)
                if overlap_size >= self.eval_params['min_overlap_voxels']:
                    predicted_overlapping_counts += 1
            results['predicted_overlapping_counts'].append(predicted_overlapping_counts)
        return dict(results)


class AccumulatedCountsEvaluation(CountsEvaluation, AccumulatedEvaluation):
    """ 
    Accumulated counts evaluation class. This class accumulates counts of 
    evaluation CountsEvaluation.
    """

    NAME = "accumulated_counts"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_accumulator()

    def reset_accumulator(self):
        """
        Resets the accumulator by clearing all stored counts.
        """
        self.accumulated_counts = defaultdict(lambda: defaultdict(list))

    def get_accumulated_evaluation(self) -> dict:
        """
        Computes metrics based on the accumulated counts.

        :return: A dictionary of accumulated metrics.
        """
        accumulated_results = defaultdict(list)
        for class_id in range(1, self.n_classes):
            true_counts = self.accumulated_counts[class_id]['true_counts']
            predicted_counts = self.accumulated_counts[class_id]['predicted_counts']
            predicted_overlapping_counts = self.accumulated_counts[class_id]['predicted_overlapping_counts']

            accumulated_results['true_counts'].append(sum(true_counts))
            accumulated_results['predicted_counts'].append(sum(predicted_counts))
            accumulated_results['predicted_overlapping_counts'].append(sum(predicted_overlapping_counts))

        return dict(accumulated_results)

    def evaluate(self, true_array: np.ndarray, predicted_array: np.ndarray) -> None:
        """
        Evaluate and accumulate counts for connected components in the given true and predicted
        segmentation arrays.

        :param true_array: The true segmentation array.
        :param predicted_array: The predicted segmentation array.
        """
        results = super().evaluate(true_array=true_array, predicted_array=predicted_array)
        for class_id in range(1, self.n_classes):
            self.accumulated_counts[class_id]['true_counts'].append(results['true_counts'][class_id - 1])
            self.accumulated_counts[class_id]['predicted_counts'].append(results['predicted_counts'][class_id - 1])
            self.accumulated_counts[class_id]['predicted_overlapping_counts'].append(results['predicted_overlapping_counts'][class_id - 1])

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

        # Format the accumulated metrics into a table
        metrics_table = self.metrics_to_table(accumulated_metrics).round(round)

        # Report the metrics table
        logger.report_table(
            title="Accumulated Counts Evaluation",
            series="Metrics",
            table_plot=metrics_table,
        )
        # Log the detailed counts table
        detailed_counts_table = self.detailed_counts_to_table().round(round)
        logger.report_table(
            title="Detailed Counts Per Iteration",
            series="Counts Details",
            table_plot=detailed_counts_table,
        )

    def detailed_counts_to_table(self) -> pd.DataFrame:
        """
        Converts the detailed counts into a pandas DataFrame for logging.

        :return: A pandas DataFrame representing the detailed counts table.
        """
        data = []
        for class_id in range(1, self.n_classes):
            iterations = len(self.accumulated_counts[class_id]['true_counts'])
            for i in range(iterations):
                row = {
                    'Class': class_id,
                    'Iteration': i + 1,
                    'True Counts': self.accumulated_counts[class_id]['true_counts'][i],
                    'Predicted Counts': self.accumulated_counts[class_id]['predicted_counts'][i],
                    'Predicted Overlapping Counts': self.accumulated_counts[class_id]['predicted_overlapping_counts'][i]
                }
                data.append(row)
        return pd.DataFrame(data)
    

    def metrics_to_table(self, metrics: dict) -> pd.DataFrame:
        """
        Converts the accumulated metrics dictionary into a pandas DataFrame for logging.

        :param metrics: The dictionary of accumulated metrics.
        :return: A pandas DataFrame representing the metrics table.
        """
        data = []
        for class_id in range(1, self.n_classes):
            row = {
                'Class': class_id,
                'True Counts': metrics['true_counts'][class_id - 1],
                'Predicted Counts': metrics['predicted_counts'][class_id - 1],
                'Predicted Overlapping Counts': metrics['predicted_overlapping_counts'][class_id - 1]
            }
            data.append(row)
        return pd.DataFrame(data)