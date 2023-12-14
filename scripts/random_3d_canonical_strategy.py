import logging
from typing import Dict, List, Union, Optional, Tuple, Any
import tensorflow as tf
from crbr.data.patch_sampling_strategies import PatchSamplingStrategy
from scipy.ndimage import label as nd_label

logger = logging.getLogger(__name__)

import numpy as np
import random
import abc


class Random3DCanonicalPatchSamplingStrategy(PatchSamplingStrategy):
    """
    Sample 3D patches from a 3D volume with desired degree of overlap and class proportions.

    Supports:
    - degree of overlap on prediction: evaluation patches can have a fraction of overlap.
    Logits are added up in overlap regions
    - class-targeted sampling: for each class, a proportion of th epatches containing
    it can be determined
    """
    NAME = "random_3d"

    def __init__(
        self,
        num_patches: int, 
        patch_size: Tuple[int, int, int],
        patch_overlap_frac: float = 0,
        class_proportions: Optional[Dict[int, float]] = None
    ):
        """
        Initializes the Random3DCanonicalPatchSamplingStrategy.

        :param num_patches: int. Number of patches to sample.
        :param patch_size: Tuple[int, int, int]. Size of the patches to sample.
        :param patch_overlap_frac: float. Fraction of patch overlap for evaluation.
        :param class_proportions: Optional[Dict[int, float]]. Proportions for each class to be included in sampling.
        """
        if not (0 <= patch_overlap_frac < 1):
            raise ValueError(f"Invalid patch_overlap_frac value, must be within [0, 1)")

        if class_proportions and not all(0 <= val <= 1 for val in class_proportions.values()):
            raise ValueError("Class proportions must be between 0 and 1")

        if class_proportions and abs(sum(class_proportions.values()) - 1) > 0.01:
            raise ValueError("Sum of class proportions must be approximately 1")

        super().__init__(
            num_patches=num_patches,
            patch_size=patch_size,
            patch_overlap_frac=patch_overlap_frac,
            class_proportions=class_proportions
        )
        self.patch_size = self.sample_params['patch_size']
        self.patch_overlap_frac = self.sample_params['patch_overlap_frac']
        self.class_proportions = self.sample_params.get('class_proportions', None)

    def _validate_patch_size(self, patch_size: Tuple[int, int, int], input_shape: List[int]):
        """
        Validates that the patch size is within the bounds of the input shape.

        :param patch_size: Tuple[int, int, int]. The size of the patch.
        :param input_shape: List[int]. The shape of the input volume.
        """
        if len(patch_size) != 3:
            raise ValueError(f"Patch size must have exactly 3 dimensions, but got {len(patch_size)} dimensions")

        is_within_bounds = all(p <= i for p, i in zip(patch_size, input_shape[:3]))
        if not is_within_bounds:
            raise ValueError(f"Patch size {patch_size} is too big for input shape {input_shape}")


    def compute_output_shape(self, input_shape: List[int]) -> List[int]:
        """
        Compute the output shape of the sampling strategy.
        :param input_shape: The input shape.
        :return: The output shape.
        """
        if len(input_shape) == 4:
            logger.warning(f"Found 4 dimensions, so considering last as channels. If not the case revise your data.")
            outshape = tuple(list(self.patch_size) + [input_shape[-1]]) # include channel dimension in output shape
        elif len(input_shape) == 3:
            outshape = tuple(self.patch_size)
        else:
            raise ValueError(f"Invalid input shape {input_shape}. Expected 3D or 4D (with channels)")
        
        self._validate_patch_size(self.patch_size, input_shape)
        return outshape  


    def _create_random_slice_tuple(self, volume_shape: List[int]) -> tuple:
        """
        Creates a random slice tuple within the bounds of the volume 
        according to patch size.

        :param volume_shape: List[int]. The shape of the volume to sample from.
        :return: tuple. A slice tuple representing a 3D patch.
        """
        x = random.randint(0, volume_shape[0] - self.patch_size[0])
        y = random.randint(0, volume_shape[1] - self.patch_size[1])
        z = random.randint(0, volume_shape[2] - self.patch_size[2])
        return (slice(x, x + self.patch_size[0]), 
                slice(y, y + self.patch_size[1]), 
                slice(z, z + self.patch_size[2]))

    def _generate_class_proportional_3Dpatches(self, volume: np.ndarray, segmentation: np.ndarray) -> List[tuple]:
        """
        Generates 3D patches ensuring class representation as specified in class_proportions.

        :param volume: np.ndarray. The volume to sample from.
        :param segmentation: np.ndarray. The segmentation mask corresponding to the volume.
        :return: List[tuple]. A list of slice tuples for patch sampling.
        """
        vol_shape = volume.shape
        combined_patches = []
        max_attempts_per_class = 50

        segmentation_int = segmentation.astype(int)
        patch_size_array = np.array(self.patch_size)
        vol_shape_spatial = np.array(vol_shape[:-1]) if len(vol_shape) == 4 else np.array(vol_shape)

        for cls, proportion in self.class_proportions.items():
            cls = int(cls)
            num_patches = int(self.sample_params['num_patches'] * proportion)

            if cls == 0:
                # For class 0, randomly select patches
                for _ in range(num_patches):
                    combined_patches.append(self._create_random_slice_tuple(volume.shape))
                continue

            # Extract the segmentation for the specific class
            class_segmentation = segmentation_int[..., cls]

            class_locations = np.argwhere(class_segmentation == 1)
            class_patches_count = 0

            for patch_num in range(num_patches):
                attempts = 0
                while attempts < max_attempts_per_class:
                    if len(class_locations) == 0:
                        break  # No locations for this class, skip to next class

                    class_loc_idx = np.random.choice(len(class_locations))
                    class_center = class_locations[class_loc_idx]

                    # Calculate start and end slices, excluding the last dimension (non-spatial)
                    class_center_spatial = class_center[:3]
                    start_slices = np.maximum(class_center_spatial - patch_size_array // 2, 0)
                    end_slices = start_slices + patch_size_array

                    # Randomly shift patch position within half the patch size
                    shift = np.random.randint(-patch_size_array // 2, patch_size_array // 2 + 1, size=3)
                    shifted_start_slices = np.maximum(start_slices + shift, 0)
                    shifted_end_slices = shifted_start_slices + patch_size_array

                    # Adjust the slices if they go beyond the volume boundaries
                    for i in range(3):
                        if shifted_end_slices[i] > vol_shape_spatial[i]:
                            overhang = shifted_end_slices[i] - vol_shape_spatial[i]
                            shifted_start_slices[i] -= overhang

                    shifted_end_slices = shifted_start_slices + patch_size_array

                    # Final check to ensure the patch is within the volume boundaries
                    if all(shifted_end_slices <= vol_shape_spatial):
                        slice_tuple = tuple(slice(start, end) for start, end in zip(shifted_start_slices, shifted_end_slices))
                        combined_patches.append(slice_tuple)
                        class_patches_count += 1
                        break
                    attempts += 1

                if attempts == max_attempts_per_class:
                    logger.warning(f"Reached max attempts for class {cls}. Could not find enough patches: {class_patches_count}")

        # Fill remaining patches with random slices if needed
        while len(combined_patches) < self.sample_params['num_patches']:
            combined_patches.append(self._create_random_slice_tuple(volume.shape))

        random.shuffle(combined_patches)
        return combined_patches


    def _generate_random_3Dslices(self, volume: np.ndarray, segmentation: np.ndarray) -> List[tuple]:
        """
        Generates 3D patches either randomly or with a proportion for each class as specified
        on class_proportions. If the total number of patches for each class does not meet num_patches, 
        the remaining patches are filled randomly.

        :param volume: np.ndarray. The volume to sample from.
        :param segmentation: np.ndarray. The segmentation mask corresponding to the volume.
        :return: List[tuple]. A list of slice tuples for patch sampling.
        """
        if self.class_proportions:
            return self._generate_class_proportional_3Dpatches(volume, segmentation)

        else:
            return [self._create_random_slice_tuple(volume.shape) for _ in range(self.sample_params['num_patches'])]


    def _create_patch(self, volume: np.ndarray, slice_tuple: tuple, org_dtype) ->  tf.Tensor:
        """
        Extracts a patch from the given volume using the specified slice tuple and converts it to a TensorFlow object.

        :param volume: np.ndarray. The volume from which to create the patch.
        :param slice_tuple: tuple. The slice tuple indicating the region of the volume to create the patch from.
        :param org_dtype. The original data type of the volume.
        :return: tf.Tensor. The created patch.
        """
        patch = volume[slice_tuple]
        return tf.convert_to_tensor(patch, dtype=org_dtype)

    def sample_patches(
        self, 
        mri: dict, 
        metadata: dict = None
    ) -> List[Dict[str, Union[tf.Tensor, str]]]:
        """
        Samples patches from the MRI volume and for the segmentation volume

        :param mri: The MRI dict to sample from.
        :param metadata: Optional metadata dict indexed by study_id keys.
        :return: A list of MRI dict patches, e.g.: [{'image': tf.Tensor, 'segmentation': tf.Tensor}]
        """
        other_keys = [key for key in mri.keys() if key not in ["image", "segmentation"]]
        mri, org_dtypes = self.convert_to_numpy_arrays(mri, ["image", "segmentation"])

        volume = mri["image"]
        segmentation = mri["segmentation"]
        self._validate_patch_size(self.patch_size, volume.shape)
        self._validate_patch_size(self.patch_size, segmentation.shape)
        
        # Sample random indices in volume in a range that allows one patch to fit
        patch_slices_list = self._generate_random_3Dslices(volume, segmentation)

        patches = []

        for patch_slice in patch_slices_list:
            patch_dict = {
                "image": self._create_patch(volume, patch_slice, org_dtypes["image"]),
                "segmentation": self._create_patch(segmentation, patch_slice, org_dtypes["segmentation"])
            }
            # Include additional metadata if present
            patch_dict.update({key: mri[key] for key in other_keys})

            patches.append(patch_dict)

        return patches

    def _pad_volume_to_fit_patches(self, volume: np.ndarray) -> np.ndarray:
        """
        Pads the given volume to ensure that it can accommodate patches of the specified size,
        considering the desired overlap between patches.

        :param volume: np.ndarray. The original volume to pad.
        :return: np.ndarray. The padded volume.
        """
        # Calculate the step size for each dimension considering the overlap
        step_x = int(self.patch_size[0] * (1 - self.patch_overlap_frac))
        step_y = int(self.patch_size[1] * (1 - self.patch_overlap_frac))
        step_z = int(self.patch_size[2] * (1 - self.patch_overlap_frac))

        # Calculate padding for each dimension once stpe ize is known
        pad_x = (step_x - (volume.shape[0] - self.patch_size[0]) % step_x) % step_x
        pad_y = (step_y - (volume.shape[1] - self.patch_size[1]) % step_y) % step_y
        pad_z = (step_z - (volume.shape[2] - self.patch_size[2]) % step_z) % step_z

        # Pad the volume
        if len(volume.shape) == 3:
            padded_volume = np.pad(volume, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='constant')
        else:
            padded_volume = np.pad(volume, ((0, pad_x), (0, pad_y), (0, pad_z), (0, 0)), mode='constant')

        return padded_volume

    def _get_deterministic_3Dpatches(self, padded_volume: np.ndarray) -> Tuple[List[np.ndarray], List[tuple]]:
        """
        Gets deterministic patches from a 3D volume, ensuring all patches have the same size and
        considering the desired overlap between them.

        :param padded_volume: np.ndarray. The 3D volume to extract patches from, already modified by self._pad_volume_to_fit_patches
        :return: List[Tuple[np.ndarray, tuple]]. A list of patches and their corresponding slice tuples.
        """

        patches = []
        slice_tuples = []

        # Calculate step size for each dimension
        step_x = int(self.patch_size[0] * (1 - self.patch_overlap_frac))
        step_y = int(self.patch_size[1] * (1 - self.patch_overlap_frac))
        step_z = int(self.patch_size[2] * (1 - self.patch_overlap_frac))

        # Ensure step size is at least 1
        step_x = max(1, step_x)
        step_y = max(1, step_y)
        step_z = max(1, step_z)

        # Iterate over the padded volume and extract patches
        for x in range(0, padded_volume.shape[0] - self.patch_size[0] + 1, step_x):
            for y in range(0, padded_volume.shape[1] - self.patch_size[1] + 1, step_y):
                for z in range(0, padded_volume.shape[2] - self.patch_size[2] + 1, step_z):
                    slice_tuple = (slice(x, x+self.patch_size[0]), slice(y, y+self.patch_size[1]), slice(z, z+self.patch_size[2]))
                    patch = padded_volume[slice_tuple]
                    patches.append(patch)
                    slice_tuples.append(slice_tuple)

        return patches, slice_tuples
    
    def get_eval_patches(
        self, mri: dict, metadata: dict = None, **kwargs
    ) -> Tuple[List[Dict[str, tf.Tensor]], List[Optional[Dict[str, Any]]]]:
        """
        Returns a list of patches of format equal to that returned by self.sample_patches as well as
        a list of patch metadata dicts, which may be used in self.combine_eval_predictions to
        reconstruct the full prediction volume. If the patch metadata dicts are not needed, a list
        of None (or any other) values of length equal to the number of patches may be returned.

        :param mri: The MRI dict to sample from.
        :param metadata: Optional metadata dict indexed by study_id keys.
        :params kwargs: Keyword arguments passed to the concrete implementation in the subclass.
        :return: A list of MRI dict patches, e.g.: [{'image': tf.Tensor, 'segmentation': tf.Tensor}]
            and a list of patch metadata dicts. The patch metadata dicts may be None if not needed.
        """
        other_keys = [key for key in mri.keys() if key not in ["image", "segmentation"]]
        mri, org_dtypes = self.convert_to_numpy_arrays(mri, ["image", "segmentation"])

        image_volume = mri["image"]
        segmentation_volume = mri["segmentation"]
        
        # Pad the volumes (depending on degree of overlap and patch size)
        # so that a patch can never go out of bounds
        padded_volume = self._pad_volume_to_fit_patches(image_volume)
        padded_segmentation_volume = self._pad_volume_to_fit_patches(segmentation_volume)

        # Extract deterministic patches and their slice tuples
        image_patches, slice_tuples = self._get_deterministic_3Dpatches(padded_volume)

        eval_patches = []
        for idx, (patch, slice_tuple) in enumerate(zip(image_patches, slice_tuples)):
            patch_dict = {
                "image": tf.convert_to_tensor(patch, dtype=org_dtypes["image"]),
                "segmentation": tf.convert_to_tensor(padded_segmentation_volume[slice_tuple], dtype=org_dtypes["segmentation"])
                
            }
            # Include additional metadata if present
            patch_dict.update({key: mri[key] for key in other_keys})

            eval_patches.append(patch_dict)

        # Return patches and metadata (slice_tuples)
        patch_metadata = [{"slice_tuple": st} for st in slice_tuples]
        
        return eval_patches, patch_metadata


    def combine_eval_patch_predictions(
        self,
        predicted_patches: List[tf.Tensor],
        patch_meta_dicts: List[Union[dict, None]],
        expected_shape: List[int]

    ) -> np.ndarray:
        """
        Adds patch predictions into a final prediction volume and pads it to match the expected shape.

        :param predicted_patches: A list of patch-wise predictions.
        :param expected_shape: The expected shape of the final prediction volume.
        :param patch_meta_dicts: A list of patch metadata dicts, each containing a 'slice_tuple' key.
        """
        # Initialize the final prediction volume with the expected shape and pad
        prediction_volume = self._pad_volume_to_fit_patches(np.zeros(expected_shape, dtype=np.float32))

        # Accumulate each patch prediction into the prediction volume
        for patch_prediction, meta_dict in zip(predicted_patches, patch_meta_dicts):
            slice_tuple = meta_dict["slice_tuple"]
            patch_prediction_np = patch_prediction

            # Update the prediction volume within the bounds defined by slice_tuple
            prediction_volume[slice_tuple] += patch_prediction_np

        # Crop prediction volume to fit the expected shape
        prediction_volume_cropped = prediction_volume[:expected_shape[0], :expected_shape[1], :expected_shape[2]]

        return prediction_volume_cropped
