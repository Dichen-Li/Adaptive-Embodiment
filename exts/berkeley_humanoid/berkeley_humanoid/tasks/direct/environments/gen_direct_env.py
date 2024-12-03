from __future__ import annotations

from berkeley_humanoid.tasks.direct import LocomotionEnv, GenDogEnvCfg
from berkeley_humanoid.tasks.direct.locomotion.locomotion_env_multi_embodi import LocomotionEnvMultiEmbodiment


class GenDirectEnv(LocomotionEnv):
    cfg: GenDogEnvCfg

    def __init__(self, cfg: GenDogEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


class GenDirectEnvMultiEmbodiment(LocomotionEnvMultiEmbodiment):
    cfg: GenDogEnvCfg

    def __init__(self, cfg: GenDogEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
