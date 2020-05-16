import torch
from allennlp.predictors import SentenceTaggerPredictor
from pathlib import Path
from BiLSTM import prepare1, prepare2, LstmTagger, PosDatasetReader, BiLSTM_predict
from build_data import preprocess
from hmm import run_hmm, hmm_viterbi
from data import load_object_from_file, logits_to_trans, compute_accuracy
from memm import memm_greedy, build_extra_decoding_arguments, run_memm
from combine_algorithms import overall_classifier, overall_choose_best_gammas, list_to_tran
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


def load_learned_data():
    train_texts, dev_texts, test_texts, sign_to_id, tran_to_id, id_to_sign, id_to_tran = preprocess()

    # Run the HMM.
    run_hmm(train_texts, dev_texts, False)
    # lambda1, lambda2 = run_hmm(train_texts, dev_texts, True)
    # dump_object_to_file((lambda1, lambda2, sign_to_id, tran_to_id, id_to_sign, id_to_tran), r"..\output\hmm_model")
    if platform.system() == "Windows":
        (lambda1, lambda2, _, _, _, _) = load_object_from_file(Path("..\output\hmm_model.pkl"))
    else:
        (lambda1, lambda2, _, _, _, _) = load_object_from_file(Path("../output/hmm_model.pkl"))

    # Restore the BiLSTM model alreay trained.
    # model, predictor = restore_model()

    # model, vocab, train_dataset, validation_dataset, cuda_device, reader = prepare1()
    # trainer, model, reader, vocab = prepare2(model, vocab, train_dataset, validation_dataset, cuda_device, reader)
    # trainer.train()
    # predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    # print("finished training")

    # dump_object_to_file(predictor, "predictor")
    if platform.system() == "Windows":
        pred_path = Path("..\output\predictor_lr_03_test_96_8.pkl")
    else:
        pred_path = Path("../output/predictor_lr_03_test_96_8.pkl")
    predictor_from_file = load_object_from_file(pred_path)

    # dump_object_to_file(model, "model")
    if platform.system() == "Windows":
        model_path = Path("..\output\model_lr_03_test_96_8.pkl")
    else:
        model_path = Path("../output/model_lr_03_test_96_8.pkl")
    model_from_file = load_object_from_file(model_path)

    # print(dev_texts)
    # print(compute_accuracy(train_texts, hmm_viterbi, 0, {}, {}, lambda1, lambda2))
    # print(compute_accuracy(dev_texts, hmm_viterbi, 0, {}, {}, lambda1, lambda2))

    # print(compute_accuracy(train_texts, BiLSTM_predict, model_from_file, predictor_from_file, sign_to_id, id_to_tran))
    # print(compute_accuracy(dev_texts, BiLSTM_predict, model_from_file, predictor_from_file, sign_to_id, id_to_tran))

    # print(build_info_sentence(sign_to_id))

    # logreg, vec, idx_to_tag_dict = run_memm(train_texts, dev_texts)
    # dump_object_to_file((logreg, vec, idx_to_tag_dict), r"..\output\memm_model")

    if platform.system() == "Windows":
        memm_path = Path("..\output\memm_model.pkl")
    else:
        memm_path = Path("../output/memm_model.pkl")
    memm_from_file = load_object_from_file(memm_path)
    (logreg, vec, idx_to_tag_dict) = memm_from_file
    extra_decoding_arguments = build_extra_decoding_arguments(train_texts)
    return lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, predictor_from_file, model_from_file, id_to_tran, sign_to_id, dev_texts


def main():
    lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, predictor_from_file, model_from_file, id_to_tran, sign_to_id, dev_texts = load_learned_data()
    gamma1 = 0.4
    gamma2 = 0.2

    #gamma1, gamma2 = overall_choose_best_gammas(lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, predictor_from_file, model_from_file, id_to_tran, sign_to_id, dev_texts[:60])

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

        overall_predicted_tags = overall_classifier(sentence, gamma1, gamma2, lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, predictor_from_file, model_from_file, id_to_tran, sign_to_id, True)
        overall_tran = list_to_tran(overall_predicted_tags)
        print("Overall transcription: ")
        print(overall_tran)


if __name__ == '__main__':
    main()
