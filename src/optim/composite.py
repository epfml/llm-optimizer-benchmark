class CompositeOptimizer:
    """Minimal optimizer facade for algorithms that use two PyTorch optimizers."""

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)
        if not self.optimizers:
            raise ValueError("CompositeOptimizer requires at least one optimizer.")
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, *args, **kwargs):
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self):
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict):
        states = state_dict["optimizers"]
        if len(states) != len(self.optimizers):
            raise ValueError(
                "Checkpoint optimizer count does not match CompositeOptimizer."
            )
        for optimizer, optimizer_state in zip(self.optimizers, states):
            optimizer.load_state_dict(optimizer_state)


class CompositeScheduler:
    def __init__(self, schedulers):
        self.schedulers = list(schedulers)

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self):
        return {"schedulers": [scheduler.state_dict() for scheduler in self.schedulers]}

    def load_state_dict(self, state_dict):
        states = state_dict["schedulers"]
        if len(states) != len(self.schedulers):
            raise ValueError(
                "Checkpoint scheduler count does not match CompositeScheduler."
            )
        for scheduler, scheduler_state in zip(self.schedulers, states):
            scheduler.load_state_dict(scheduler_state)


def split_muon_param_groups(group_specs, fallback_param_ids):
    """Partition groups between PyTorch Muon and an AdamW fallback."""
    muon_groups = []
    adamw_groups = []
    for group in group_specs:
        shared = {key: value for key, value in group.items() if key != "params"}
        muon_params = [
            parameter
            for parameter in group["params"]
            if parameter.ndim == 2 and id(parameter) not in fallback_param_ids
        ]
        adamw_params = [
            parameter
            for parameter in group["params"]
            if parameter.ndim != 2 or id(parameter) in fallback_param_ids
        ]
        if muon_params:
            muon_groups.append({**shared, "params": muon_params})
        if adamw_params:
            adamw_groups.append({**shared, "params": adamw_params})
    if not muon_groups or not adamw_groups:
        raise ValueError(
            "muon-pytorch requires both hidden 2D matrices and AdamW fallback "
            "parameters. Check the model parameter partition."
        )
    return muon_groups, adamw_groups
