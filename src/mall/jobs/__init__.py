"""High-level jobs of the project."""

# %% IMPORTS

from mall.jobs.inference import InferenceJob
from mall.jobs.promotion import PromotionJob
from mall.jobs.training import TrainingJob
from mall.jobs.tuning import TuningJob

# %% TYPES

JobKind = TuningJob | TrainingJob | PromotionJob | InferenceJob

# %% EXPORTS

__all__ = ["TuningJob", "TrainingJob", "PromotionJob", "InferenceJob", "JobKind"]
