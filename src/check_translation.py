import os
from BiLSTM import prepare1, prepare2, LstmTagger, PosDatasetReader
from build_data import preprocess, break_into_sentences
from hmm import run_hmm, hmm_viterbi, hmm_compute_accuracy
from data import load_object_from_file, logits_to_trans
from memm import memm_greedy, build_extra_decoding_arguments, run_memm
from parse_json import parse_json
from pathlib import Path
from get_texts_details import get_dialect
import statistics
import platform


def parsed_json_to_HMM_format(sentences, sign_to_id):
    HMM_sentences = {}

    for key in sentences:
        list = []
        for _, _, _, sign in sentences[key]:
            list.append((sign, ""))

        HMM_sentences[key] = list

    return HMM_sentences


def parsed_json_to_allen_format(sentences, sign_to_id):
    allen_sentences = {}

    for key in sentences:
        signs = ""
        for _, _, _, sign in sentences[key]:
            try:
                signs += str(sign_to_id[sign]) + " "
            except:
                signs += "0 " # default index

        allen_sentences[key] = signs

    return allen_sentences


def compute_accuracy(sentences, predicted_tags):
    correct = 0
    total = 0

    for key in sentences:
        for i in range(len(sentences[key])):
            total += 1
            c = sentences[key][i][1] + sentences[key][i][2] if not sentences[key][i][2] is None else sentences[key][i][1]
            if c == predicted_tags[key][i]:
                correct += 1

    if total != 0:
        return correct / float(total)
    else:
        return "None"


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


def make_algorithm_prediction(algorithm, sentences, format_function, sign_to_id, prediction_function, *args):
    formated_sentences = format_function(sentences, sign_to_id)
    predicted_tags = {}
    for key in formated_sentences:
        line_predicted_tags = prediction_function(formated_sentences[key], *args)
        predicted_tags[key] = line_predicted_tags

    #print(algorithm + "predictions: " + str(predicted_tags))
    accuracy = compute_accuracy(sentences, predicted_tags)
    print(algorithm + " percentage: " + str(accuracy))
    return accuracy


def biLSTM_predict(line, id_to_tran, predictor_from_file, model_from_file):
    tag_logits = predictor_from_file.predict(line)['tag_logits']
    biLSTM_predicted_tags, biLSTM_predicted2_tags, biLSTM_predicted3_tags, _, _, _ = logits_to_trans(tag_logits,
                model_from_file, id_to_tran)
    return biLSTM_predicted_tags


def make_predictions(sentences, lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, sign_to_id,
                        id_to_tran, predictor_from_file, model_from_file):
    HMM_accuracy = make_algorithm_prediction("HMM", sentences, parsed_json_to_HMM_format, sign_to_id, hmm_viterbi, 0, {}, {}, {}, {},
                              {}, lambda1, lambda2)

    MEMM_accuracy = make_algorithm_prediction("MEMM", sentences, parsed_json_to_HMM_format, sign_to_id, memm_greedy, logreg, vec,
                              idx_to_tag_dict, extra_decoding_arguments)

    biLSTM_accuracy = make_algorithm_prediction("biLSTM", sentences, parsed_json_to_allen_format, sign_to_id, biLSTM_predict, id_to_tran,
                              predictor_from_file, model_from_file)

    return HMM_accuracy, MEMM_accuracy, biLSTM_accuracy


def operate_on_file(directory, corpus, file, lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments,
                    sign_to_id, id_to_tran, predictor_from_file, model_from_file):
    print(file)
    f = directory / corpus / file
    parsed = parse_json(f)

    dict = {}
    dict[file] = parsed
    sentences = break_into_sentences(dict)
    #print(sentences)
    HMM_accuracy, MEMM_accuracy, biLSTM_accuracy = make_predictions(sentences, lambda1, lambda2, logreg, vec,
            idx_to_tag_dict, extra_decoding_arguments, sign_to_id, id_to_tran, predictor_from_file, model_from_file)

    dialect = get_dialect(corpus, file)
    print(dialect)

    if biLSTM_accuracy == 'None':
        return

    global dialects_HMM
    global dialects_MEMM
    global dialects_biLSTM
    if dialect not in dialects_HMM:
        dialects_HMM[dialect] = [HMM_accuracy]
        dialects_MEMM[dialect] = [MEMM_accuracy]
        dialects_biLSTM[dialect] = [biLSTM_accuracy]
    else:
        dialects_HMM[dialect].append(HMM_accuracy)
        dialects_MEMM[dialect].append(MEMM_accuracy)
        dialects_biLSTM[dialect].append(biLSTM_accuracy)

    #print(dialects_HMM)
    #print(dialects_MEMM)
    print(dialects_biLSTM)


def compute_averages():
    for dialect in dialects_HMM:
        print(dialect + " HMM average: " + str(statistics.mean(dialects_HMM[dialect])))
        print(dialect + " MEMM average: " + str(statistics.mean(dialects_MEMM[dialect])))
        print(dialect + " biLSTM average: " + str(statistics.mean(dialects_biLSTM[dialect])))


def main():
    directory = Path(r"../raw_data/test_texts")
    global dialects_HMM
    dialects_HMM = {}
    global dialects_MEMM
    dialects_MEMM = {}
    global dialects_biLSTM
    dialects_biLSTM = {}

    lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, sign_to_id, id_to_tran, \
    predictor_from_file, model_from_file = copied_code_from_translate_Akkadian()

    #for file in os.listdir(directory / "random"):
    #    operate_on_file(directory / "random", file, lambda1, lambda2, logreg, vec, idx_to_tag_dict,
    #                    extra_decoding_arguments, sign_to_id, id_to_tran, predictor_from_file, model_from_file)

    corpus = "riao"
    for file in os.listdir(directory / corpus):
        operate_on_file(directory, corpus, file, lambda1, lambda2, logreg, vec, idx_to_tag_dict,
                        extra_decoding_arguments, sign_to_id, id_to_tran, predictor_from_file, model_from_file)

    compute_averages()


if __name__ == '__main__':
    main()
