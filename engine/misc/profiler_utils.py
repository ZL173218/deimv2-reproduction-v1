"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from calflops import calculate_flops
from typing import Tuple


def _resolve_hw(value):
    """Return a (height, width) tuple given a scalar/list value."""

    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return int(value[0]), int(value[1])
        if len(value) == 1:
            scalar = int(value[0])
            return scalar, scalar
    elif value is not None:
        scalar = int(value)
        return scalar, scalar
    return None


def _synchronize_eval_spatial_size(model, height: int, width: int) -> None:
    """Ensure eval-only positional caches match the dummy input size."""

    for module in model.modules():
        if hasattr(module, 'eval_spatial_size') and module.eval_spatial_size is not None:
            current = module.eval_spatial_size
            target = [height, width]
            if list(current) != target:
                module.eval_spatial_size = target
                if hasattr(module, '_reset_parameters'):
                    module._reset_parameters()


def stats(
    cfg,
    input_shape: Tuple = (1, 3, 640, 640),
) -> Tuple[int, dict]:

    eval_hw = _resolve_hw(cfg.yaml_cfg.get('eval_spatial_size'))
    if eval_hw is None:
        train_collate = (
            cfg.yaml_cfg
            .get('train_dataloader', {})
            .get('collate_fn', {})
        )
        eval_hw = _resolve_hw(train_collate.get('base_size'))

    if eval_hw is None:
        height, width = input_shape[2], input_shape[3]
    else:
        height, width = eval_hw


    input_shape = (input_shape[0], input_shape[1], height, width)

    model_for_info = copy.deepcopy(cfg.model).deploy()
    _synchronize_eval_spatial_size(model_for_info, height, width)

    flops, macs, _ = calculate_flops(
        model=model_for_info,
        input_shape=input_shape,
        output_as_string=True,
        output_precision=4,
        print_detailed=False,
    )
    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info

    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, params)}