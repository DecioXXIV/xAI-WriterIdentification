CLASSIFIERS = ["ResNet18", "ViT_SwinTiny"]
SCHEDULERS = ["cos_annealing", "reduce_lr_on_plateau"]
FT_MODES = ["frozen", "full"]


__all__ = [CLASSIFIERS, SCHEDULERS, FT_MODES]