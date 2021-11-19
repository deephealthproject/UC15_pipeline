"""
Module with utilities for the training pipeline.
"""
import pytorch_lightning.callbacks as callbacks


class FeatureExtractorFreezeUnfreeze(callbacks.BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=1):
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # Freeze the feature extractor module
        print("Going to freeze the feature extractor")
        # self.freeze(pl_module.feature_extractor)  # Not working properly
        for param in pl_module.model.feature_extractor.parameters():
            param.requires_grad = False

    def finetune_function(self, pl_module, current_epoch,
                          optimizer, optimizer_idx):
        # If we reached the desired epoch, unfreeze the feature extractor
        if current_epoch == self.unfreeze_at_epoch:
            print("Going to unfreeze the feature extractor")
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.feature_extractor,
                optimizer=optimizer,
                train_bn=True,
                initial_denom_lr=1.0
            )
