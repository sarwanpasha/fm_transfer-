"""Make cpa-tools 0.8.x work with Lightning >= 2 (training_epoch_end/validation_epoch_end were removed).
Usage: python fm_cpa_patch.py <site-packages>/cpa/_task.py"""
import sys
p = sys.argv[1]; s = open(p).read()
if "_FM_LIGHTNING2_PATCH" not in s:
    s += '''

# _FM_LIGHTNING2_PATCH: Lightning >= 2 removed *_epoch_end(outputs); collect step outputs and call the original methods.
def _fm_patch(cls):
    tee, vee = cls.__dict__["training_epoch_end"], cls.__dict__["validation_epoch_end"]
    ts, vs = cls.__dict__["training_step"], cls.__dict__["validation_step"]
    delattr(cls, "training_epoch_end"); delattr(cls, "validation_epoch_end")
    def training_step(self, batch, batch_idx):
        out = ts(self, batch, batch_idx); self.__dict__.setdefault("_fm_tr", []).append(out); return out
    def validation_step(self, batch, batch_idx):
        out = vs(self, batch, batch_idx); self.__dict__.setdefault("_fm_va", []).append(out); return out
    def on_train_epoch_end(self):
        tee(self, self.__dict__.get("_fm_tr", [])); self._fm_tr = []
    def on_validation_epoch_end(self):
        vee(self, self.__dict__.get("_fm_va", [])); self._fm_va = []
    cls.training_step, cls.validation_step = training_step, validation_step
    cls.on_train_epoch_end, cls.on_validation_epoch_end = on_train_epoch_end, on_validation_epoch_end

_fm_patch(CPATrainingPlan)
'''
    open(p, "w").write(s); print("patched")
