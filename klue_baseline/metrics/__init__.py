from .base import BaseMetric, LabelRequiredMetric
from .functional import (
    klue_dp_las_macro_f1,
    klue_dp_las_micro_f1,
    klue_dp_uas_macro_f1,
    klue_dp_uas_micro_f1,
    klue_mrc_em,
    klue_mrc_rouge_w,
    klue_ner_char_macro_f1,
    klue_ner_entity_macro_f1,
    klue_nli_acc,
    klue_re_auprc,
    klue_re_micro_f1,
    klue_sts_f1,
    klue_sts_pearsonr,
    wos_jga,
    wos_slot_micro_f1,
    ynat_macro_f1,
)

YNAT_MacroF1 = BaseMetric(ynat_macro_f1)
KlueNLI_ACC = BaseMetric(klue_nli_acc)
KlueSTS_Pearsonr = BaseMetric(klue_sts_pearsonr)
KlueSTS_F1 = BaseMetric(klue_sts_f1)
KlueNER_CharMacroF1 = LabelRequiredMetric(klue_ner_char_macro_f1)
KlueNER_EntityMacroF1 = LabelRequiredMetric(klue_ner_entity_macro_f1)
KlueRE_MicroF1 = LabelRequiredMetric(klue_re_micro_f1)
KlueRE_AUPRC = BaseMetric(klue_re_auprc)
KlueDP_UASMacroF1 = BaseMetric(klue_dp_uas_macro_f1)
KlueDP_UASMicroF1 = BaseMetric(klue_dp_uas_micro_f1)
KlueDP_LASMacroF1 = BaseMetric(klue_dp_las_macro_f1)
KlueDP_LASMicroF1 = BaseMetric(klue_dp_las_micro_f1)
KlueMRC_EM = BaseMetric(klue_mrc_em)
KlueMRC_ROUGEW = BaseMetric(klue_mrc_rouge_w)
WoS_JGA = BaseMetric(wos_jga)
WoS_SlotMicroF1 = BaseMetric(wos_slot_micro_f1)
