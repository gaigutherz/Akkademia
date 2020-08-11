r"""akkadian <https://github.com/gaigutherz/Translating-Akkadian-using-NLP> is a tool for transliterating akkadian.

Transliterating akkadian signs::

    >>> import akkadian.transliterate as akk
    >>> print(akk.transliterate("𒁹𒀭𒌍𒋀𒈨𒌍𒌷𒁀"))
    "{m}-{d}-30-ŠEŠ.MEŠ {URU}-ba-"

Transliterating akkadian signs using BiLSTM::

    >>> import akkadian.transliterate as akk
    >>> print(akk.transliterate_bilstm("𒁹𒀭𒌍𒋀𒈨𒌍𒌷𒁀"))
    "{m}-{d}-30-ŠEŠ.MEŠ {URU}-ba-"

Top three options of transliterating akkadian signs using BiLSTM::

    >>> import akkadian.transliterate as akk
    >>> print(akk.transliterate_bilstm_top3("𒁹𒀭𒌍𒋀𒈨𒌍𒌷𒁀"))
    ('{m}-{d}-30-ŠEŠ.MEŠ {URU}-ba-', 'ana 30 PAP.MEŠ-30 URU BA-', '1-AN.GIŠ.BARA₂.ME-eš URU-ba ')

Transliterating akkadian signs using MEMM::

    >>> import akkadian.transliterate as akk
    >>> print(akk.transliterate_memm("𒁹𒀭𒌍𒋀𒈨𒌍𒌷𒁀"))
    "{m}-{d}-MAŠ-GU₂.MEŠ {URU}-ba-"

Transliterating akkadian signs using HMM::

    >>> import akkadian.transliterate as akk
    >>> print(akk.transliterate_hmm("𒁹𒀭𒌍𒋀𒈨𒌍𒌷𒁀"))
    "{m}-{d}-30-ŠEŠ.MEŠ-eri-ba"
"""

import os

this_dir, this_filename = os.path.split(__file__)
output_path = os.path.join(this_dir, "output")

hmm_path = os.path.join(output_path, "hmm_model.pkl")
memm_path = os.path.join(output_path, "memm_model.pkl")
if os.name == "nt":
    bilstm_path = os.path.join(output_path, "bilstm_model_windows.pkl")
else:
    bilstm_path = os.path.join(output_path, "bilstm_model_linux.pkl")

dictionary_path = os.path.join(output_path, "dictionary.txt")

train_path = os.path.join("..", "BiLSTM_input", "allen_train_texts.txt")
validation_path = os.path.join("..", "BiLSTM_input", "allen_dev_texts.txt")

predictor_path = os.path.join("output", "predictor")
model_path = os.path.join("output", "model")
