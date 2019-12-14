import torch
import numpy as np
import time
from allennlp.predictors import SentenceTaggerPredictor
from pathlib import Path

import pickle

#from build_data import local_path
from BiLSTM import prepare1, prepare2, LstmTagger, PosDatasetReader
from build_data import preprocess
from hmm import run_hmm, hmm_viterbi, hmm_compute_accuracy
from data import load_object_from_file, logits_to_trans
from memm import memm_greedy, build_extra_decoding_arguments, run_memm
import platform

SIGNS_IN_LINE = 10


def restore_model():
    model, vocab, train_dataset, validation_dataset, cuda_device, reader = prepare1()
    with open("model_200emd_200hid_1batch.th", "rb") as f:
        model.load_state_dict(torch.load(f, map_location="cpu"))

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

    return model, predictor


def sentence_to_allen_format(sentence, sign_to_id, usingRealSigns):
    signs = ""

    if usingRealSigns:
        for sign in sentence:
            if sign == " " or sign == "\t" or sign == "\n":
                continue
            try:
                signs += str(sign_to_id[sign]) + " "
            except:
                signs += "0 " # default index

    else:
        for sign in sentence.split(","):
            signs += sign + " "

    return signs


def build_info_sentence(sign_to_id):
    sent = "Which signs would you like to translate into transcriptions today? :)\n"

    i = 0
    for sign in sign_to_id:
        sent += "for " + sign + " use " + str(sign_to_id[sign]) + "; "
        i += 1
        if i % SIGNS_IN_LINE == 0:
            sent += "\n"

    sent += "\nEnter numbers seperated with ','. For example: '1,43,37'.\n"

    return sent


def sentence_to_HMM_format(sentence):
    list = []
    for sign in sentence:
        if sign == " " or sign == "\t" or sign == "\n":
            continue
        list.append((sign, ""))

    return list


def list_to_tran(list):
    transcription = ""
    for tran in list:
        if tran[-3:] == "(0)":
            transcription += tran[:-3]
        elif tran[-4:] == "(0)-" or tran[-4:] == "(0).":
            transcription += tran[:-4] + tran[-1]
        elif tran[-1] == ")" or tran[-2:] == ")-" or tran[-2:] == ").":
            continue
        else:
            transcription += tran

        if tran[-1] != "-" and tran[-1] != ".":
            transcription += " "

    return transcription


def main():
    start_time = time.time()
    train_texts, dev_texts, sign_to_id, tran_to_id, id_to_sign, id_to_tran = preprocess()

    # Run the HMM.
    run_hmm(train_texts, dev_texts, False)
    # lambda1, lambda2 = run_hmm(train_texts, dev_texts, True)
    # dump_object_to_file((lambda1, lambda2, sign_to_id, tran_to_id, id_to_sign, id_to_tran), r"..\output\hmm_model")
    if platform.system() == "Windows":
        (lambda1, lambda2, _, _, _, _) = load_object_from_file(Path("..\output\hmm_model.pkl"))
    else:
        (lambda1, lambda2, _, _, _, _) = load_object_from_file(Path("../output/hmm_model.pkl"))

    # Restore the BiLSTM model alreay trained.
    #model, predictor = restore_model()

    #model, vocab, train_dataset, validation_dataset, cuda_device, reader = prepare1()
    #trainer, model, reader, vocab = prepare2(model, vocab, train_dataset, validation_dataset, cuda_device, reader)
    #trainer.train()
    #predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    #print("finished training")

    #dump_object_to_file(predictor, "predictor")
    if platform.system() == "Windows":
        pred_path = Path("..\output\predictor_lr_03_test_96_8.pkl")
    else:
        pred_path = Path("../output/predictor_lr_03_test_96_8.pkl")
    predictor_from_file = load_object_from_file(pred_path)

    #dump_object_to_file(model, "model")
    if platform.system() == "Windows":
        model_path = Path("..\output\model_lr_03_test_96_8.pkl")
    else:
        model_path = Path("../output/model_lr_03_test_96_8.pkl")
    model_from_file = load_object_from_file(model_path)

    #print(dev_texts)
    #print(hmm_compute_accuracy(train_texts, lambda1, lambda2))
    #print(hmm_compute_accuracy(dev_texts, lambda1, lambda2))

    #print(BiLSTM_compute_accuracy(train_texts, model_from_file, predictor_from_file, sign_to_id, id_to_tran))
    #print(BiLSTM_compute_accuracy(dev_texts, model_from_file, predictor_from_file, sign_to_id, id_to_tran))
    #exit()

    #print(build_info_sentence(sign_to_id))

    # logreg, vec, idx_to_tag_dict = run_memm(train_texts, dev_texts)
    # dump_object_to_file((logreg, vec, idx_to_tag_dict), r"..\output\memm_model")

    if platform.system() == "Windows":
        memm_path = Path("..\output\memm_model.pkl")
    else:
        memm_path = Path("../output/memm_model.pkl")
    memm_from_file = load_object_from_file(memm_path)

    (logreg, vec, idx_to_tag_dict) = memm_from_file

    extra_decoding_arguments = build_extra_decoding_arguments(train_texts)

    '''
    Sennacherib = "𒁹𒀭𒌍𒋀𒈨𒌍𒌷𒁀"
    sentence = ""
    for sign in Sennacherib:
        sentence += str(sign_to_id[sign]) + ","
    print("Sennacherib is " + sentence)
    '''

    while True:
        sentence = input("write here:")

        if sentence == "":
            continue

        HMM_predicted_tags = hmm_viterbi(sentence_to_HMM_format(sentence), 0, {}, {}, {}, {}, {}, lambda1, lambda2)
        print("HMM prediction: ")
        print(HMM_predicted_tags)

        MEMM_predicted_tags = memm_greedy(sentence_to_HMM_format(sentence), logreg, vec, idx_to_tag_dict, extra_decoding_arguments)
        print("MEMM prediction: ")
        print(MEMM_predicted_tags)

        # BiLSTM prediction
        tag_logits = predictor_from_file.predict(sentence_to_allen_format(sentence, sign_to_id, True))['tag_logits']
        biLSTM_predicted_tags, biLSTM_predicted2_tags, biLSTM_predicted3_tags = logits_to_trans(tag_logits, model_from_file, id_to_tran)
        print("biLSTM prediction: ")
        print(biLSTM_predicted_tags)

        print("biLSTM second option prediction: ")
        print(biLSTM_predicted2_tags)

        print("biLSTM third option prediction: ")
        print(biLSTM_predicted3_tags)

        HMM_tran = list_to_tran(HMM_predicted_tags)
        print("HMM transcription: ")
        print(HMM_tran)

        MEMM_tran = list_to_tran(MEMM_predicted_tags)
        print("MEMM transcription: ")
        print(MEMM_tran)

        biLSTM_tran = list_to_tran(biLSTM_predicted_tags)
        print("biLSTM transcription: ")
        print(biLSTM_tran)


if __name__ == '__main__':
    main()
