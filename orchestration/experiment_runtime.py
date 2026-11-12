

from data.experiment import ExperimentResult, ExperimentSpec
from orchestration.runner import ExperimentRunner
from simulation_interface.embodied_environment import EmbodiedEnvironment


class ExperimentRuntime:

    def __init__(
        self,
        runner: ExperimentRunner,
        environment: EmbodiedEnvironment,
    ):
        self._runner = runner
        self._environment = environment

    def run(
        self,
        experiment: ExperimentSpec,
    ) -> ExperimentResult:
        return self._runner.run(experiment)

    def close(self) -> None:
        self._environment.close()