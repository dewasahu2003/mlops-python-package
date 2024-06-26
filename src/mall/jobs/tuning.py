"""Define a job for finding the best hyperparameters for a model."""

# %% IMPORTS

import typing as T

import pydantic as pdt

from mall.core import metrics, models, schemas
from mall.io import datasets, services
from mall.jobs import base
from mall.utils import searchers, splitters

# %% JOBS


class TuningJob(base.Job):
    """Find the best hyperparameters for a model.

    Parameters:
        run_config (services.MlflowService.RunConfig): mlflow run config.
        inputs (datasets.ReaderKind): reader for the inputs data.
        targets (datasets.ReaderKind): reader for the targets data.
        model (models.ModelKind): machine learning model to tune.
        metric (metrics.MetricKind): tuning metric to optimize.
        splitter (splitters.SplitterKind): data sets splitter.
        searcher: (searchers.SearcherKind): hparams searcher.
    """

    KIND: T.Literal["TuningJob"] = "TuningJob"

    # Run
    run_config: services.MlflowService.RunConfig = services.MlflowService.RunConfig(name="Tuning")
    # Data
    inputs: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    targets: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    # Model
    model: models.ModelKind = pdt.Field(models.BaselineSklearnModel(), discriminator="KIND")
    # Metric
    metric: metrics.MetricKind = pdt.Field(metrics.SklearnMetric(), discriminator="KIND")
    # splitter
    splitter: splitters.SplitterKind = pdt.Field(
        splitters.TrainTestSplitter(), discriminator="KIND"
    )
    # Searcher
    searcher: searchers.SearcherKind = pdt.Field(
        searchers.GridCVSearcher(
            param_grid={
                "max_depth": [3, 5, 7],
            }
        ),
        discriminator="KIND",
    )

    @T.override
    def run(self) -> base.Locals:
        """Run the tuning job in context."""
        # services
        # - logger
        logger = self.logger_service.logger()
        logger.info("With logger: {}", logger)
        # - mlflow
        with self.mlflow_service.run_context(run_config=self.run_config) as run:
            logger.info("With mlflow run id: {}", run.info.run_id)
            # data
            # - inputs
            logger.info("Read inputs: {}", self.inputs)
            inputs_ = self.inputs.read()  # unchecked!
            inputs = schemas.InputsSchema.check(inputs_)
            logger.debug("- Inputs shape: {}", inputs.shape)
            # - targets
            logger.info("Read targets: {}", self.targets)
            targets_ = self.targets.read()  # unchecked!
            targets = schemas.TargetsSchema.check(targets_)
            logger.debug("- Targets shape: {}", targets.shape)
            # model
            logger.info("With model: {}", self.model)
            # metric
            logger.info("With metric: {}", self.metric)
            # splitter
            logger.info("With splitter: {}", self.splitter)
            # searcher
            logger.info("Run searcher: {}", self.searcher)
            results, best_score, best_params = self.searcher.search(
                model=self.model,
                metric=self.metric,
                inputs=inputs,
                targets=targets,
                cv=self.splitter,
            )
            logger.debug("- Results: {}", results.shape)
            logger.debug("- Best Score: {}", best_score)
            logger.debug("- Best Params: {}", best_params)
        return locals()
