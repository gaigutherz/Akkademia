import os
from BiLSTM import prepare1, prepare2, LstmTagger, PosDatasetReader
from build_data import preprocess
from hmm import run_hmm, hmm_viterbi, hmm_compute_accuracy
from data import load_object_from_file, logits_to_trans
from memm import memm_greedy, build_extra_decoding_arguments, run_memm
from parse_json import parse_json
from pathlib import Path
import platform


def parsed_json_to_HMM_format(parsed):
    list = []
    for _, _, _, sign in parsed:
        list.append((sign, ""))

    return list


def parsed_json_to_allen_format(parsed, sign_to_id):
    signs = ""

    for _, _, _, sign in parsed:
        try:
            signs += str(sign_to_id[sign]) + " "
        except:
            signs += "0 " # default index

    return signs


def compute_accuracy(parsed, output):
    correct = 0
    total = 0
    for i in range(len(parsed)):
        total += 1
        c = parsed[i][1] + parsed[i][2] if not parsed[i][2] is None else parsed[i][1]
        if c == output[i]:
            correct += 1

    return correct / float(total)


def copied_code_from_translate_Akkadian():
    train_texts, dev_texts, sign_to_id, tran_to_id, id_to_sign, id_to_tran = preprocess()

    # Run the HMM.
    run_hmm(train_texts, dev_texts, False)
    # lambda1, lambda2 = run_hmm(train_texts, dev_texts, True)
    (lambda1, lambda2, _, _, _, _) = load_object_from_file(Path("../output/hmm_model.pkl"))

    memm_path = Path("../output/memm_model.pkl")
    memm_from_file = load_object_from_file(memm_path)

    (logreg, vec, idx_to_tag_dict) = memm_from_file

    extra_decoding_arguments = build_extra_decoding_arguments(train_texts)

    #dump_object_to_file(predictor, "predictor")
    pred_path = Path("../output/predictor_lr_03_test_96_8.pkl")
    predictor_from_file = load_object_from_file(pred_path)

    #dump_object_to_file(model, "model")

    model_path = Path("../output/model_lr_03_test_96_8.pkl")
    model_from_file = load_object_from_file(model_path)

    return lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, sign_to_id, id_to_tran, \
           predictor_from_file, model_from_file


def make_prediction(parsed, lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, sign_to_id, \
                        id_to_tran, predictor_from_file, model_from_file):
    HMM_predicted_tags = hmm_viterbi(parsed_json_to_HMM_format(parsed), 0, {}, {}, {}, {}, {}, lambda1, lambda2)
    #print("HMM prediction: " + str(HMM_predicted_tags))
    print("HMM precentage: " + str(compute_accuracy(parsed, HMM_predicted_tags)))

    #MEMM_predicted_tags = memm_greedy(parsed_json_to_HMM_format(parsed), logreg, vec, idx_to_tag_dict,
    #                                  extra_decoding_arguments)
    #print("MEMM prediction: " + str(MEMM_predicted_tags))
    #print("MEMM precentage: " + str(compute_accuracy(parsed, MEMM_predicted_tags)))

    # BiLSTM prediction
    tag_logits = predictor_from_file.predict(parsed_json_to_allen_format(parsed, sign_to_id))['tag_logits']
    biLSTM_predicted_tags, biLSTM_predicted2_tags, biLSTM_predicted3_tags = logits_to_trans(tag_logits, model_from_file,
                                                                                            id_to_tran)
    #print("BiLSTM prediction: " + str(biLSTM_predicted_tags))
    print("BiLSTM precentage: " + str(compute_accuracy(parsed, biLSTM_predicted_tags)))


def operate_on_file(directory, file, lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments,
                    sign_to_id, id_to_tran, predictor_from_file, model_from_file):
    print(file)
    f = directory / file
    parsed = parse_json(f)
    # print(parsed)
    make_prediction(parsed, lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, sign_to_id,
                    id_to_tran, predictor_from_file, model_from_file)


def main():
    directory = Path(r"../raw_data/test_texts")

    lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, sign_to_id, id_to_tran, \
    predictor_from_file, model_from_file = copied_code_from_translate_Akkadian()

    for file in os.listdir(directory / "random"):
        operate_on_file(directory / "random", file, lambda1, lambda2, logreg, vec, idx_to_tag_dict,
                        extra_decoding_arguments, sign_to_id, id_to_tran, predictor_from_file, model_from_file)

    #for file in os.listdir(directory / "riao"):
    #    operate_on_file(directory / "riao", file, lambda1, lambda2, logreg, vec, idx_to_tag_dict,
    #                    extra_decoding_arguments, sign_to_id, id_to_tran, predictor_from_file, model_from_file)


if __name__ == '__main__':
    main()
