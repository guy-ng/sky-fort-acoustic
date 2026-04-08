# efficientat_mn10_v6.pt — Head Shape & Class Count

Inspected: 2026-04-07
Source: models/efficientat_mn10_v6.pt

## Head Shape
- num_classes: 1
- classifier.weight.shape: (1, 1280)  (final layer: classifier.5.weight)
- classifier.bias.shape: (1,)
- penultimate: classifier.2.weight (1280, 960)
- total_params: 4227471 (~4.23M)

## Top-level state_dict keys
OrderedDict of torch tensors. Representative keys:
- features.0.0.weight … features.16.1.* (MobileNetV3 backbone)
- features.{4,5,6,11,12,13,14,15}.block.2.conc_se_layers.0.{fc1,fc2}.{weight,bias} (SE attention)
- classifier.2.weight (1280, 960)
- classifier.2.bias (1280,)
- classifier.5.weight (1, 1280)
- classifier.5.bias (1,)

Checkpoint is a bare `OrderedDict` state_dict (no `state_dict` / `model_state_dict` wrapper).

## Notes for downstream plans
- Plan 21-03 (ONNX conversion): export shape (1, 1, 128, T_FRAMES) -> (1, 1). Single-logit head.
- Plan 21-05 (inference): output tensor shape is (batch, 1). Apply `sigmoid` for drone probability.
- Since num_classes == 1, this is a **binary sigmoid head** (drone / not-drone). D-11 `per_class_thresholds` collapses to a single `enter_threshold` + `exit_threshold` pair.
- Hysteresis state machine operates on scalar probability, not argmax.
