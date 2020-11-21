""" klue_baseline package info """
__version__ = "0.1.0"
__author__ = "KLUE project contributors"


from klue_baseline.data import (
    KlueDPProcessor,
    KlueMRCProcessor,
    KlueNERProcessor,
    KlueNLIProcessor,
    KlueREProcessor,
    KlueSTSProcessor,
    WoSProcessor,
    YNATProcessor,
)
from klue_baseline.metrics import (
    KlueDP_LASMacroF1,
    KlueDP_LASMicroF1,
    KlueDP_UASMacroF1,
    KlueDP_UASMicroF1,
    KlueMRC_EM,
    KlueMRC_ROUGEW,
    KlueNER_CharMacroF1,
    KlueNER_EntityMacroF1,
    KlueNLI_ACC,
    KlueRE_AUPRC,
    KlueRE_MicroF1,
    KlueSTS_F1,
    KlueSTS_Pearsonr,
    WoS_JGA,
    WoS_SlotMicroF1,
    YNAT_MacroF1,
)
from klue_baseline.models import (
    DPTransformer,
    DSTTransformer,
    MRCTransformer,
    NERTransformer,
    RETransformer,
    SCTransformer,
    STSTransformer,
)
from klue_baseline.task import KlueTask

# Register Task - KlueTask(processor, model_type, metrics)
KLUE_TASKS = {
    "ynat": KlueTask(YNATProcessor, SCTransformer, {"macro_f1": YNAT_MacroF1}),
    "klue-nli": KlueTask(KlueNLIProcessor, SCTransformer, {"accuracy": KlueNLI_ACC}),
    "klue-sts": KlueTask(KlueSTSProcessor, STSTransformer, {"pearsonr": KlueSTS_Pearsonr, "f1": KlueSTS_F1}),
    "klue-ner": KlueTask(
        KlueNERProcessor,
        NERTransformer,
        {"entity_macro_f1": KlueNER_EntityMacroF1, "character_macro_f1": KlueNER_CharMacroF1},
    ),
    "klue-re": KlueTask(KlueREProcessor, RETransformer, {"micro_f1": KlueRE_MicroF1, "auprc": KlueRE_AUPRC}),
    "klue-dp": KlueTask(
        KlueDPProcessor,
        DPTransformer,
        {
            "uas_macro_f1": KlueDP_UASMacroF1,
            "uas_micro_f1": KlueDP_UASMicroF1,
            "las_macro_f1": KlueDP_LASMacroF1,
            "las_micro_f1": KlueDP_LASMicroF1,
        },
    ),
    "klue-mrc": KlueTask(KlueMRCProcessor, MRCTransformer, {"exact_match": KlueMRC_EM, "rouge_w": KlueMRC_ROUGEW}),
    "wos": KlueTask(
        WoSProcessor,
        DSTTransformer,
        {"joint_goal_acc": WoS_JGA, "slot_micro_f1": WoS_SlotMicroF1},
    ),
}
