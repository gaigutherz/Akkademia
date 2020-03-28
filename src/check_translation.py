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
from combine_algorithms import combine_tags
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


def is_equal(original, prediction):
    if original == prediction:
        return True
    if original[:-1] == prediction[:-1] and original[-1] in ".-" and prediction[-1] in ".-":
        return True
    return False


def is_equal_without_segmentation(original, prediction):
    if original[:-1] == prediction and original[-1] in ".-":
        return True
    elif original == prediction[:-1] and prediction[-1] in ".-":
        return True
    return False


def to_canonical_rep(tran):
    if tran[0] == "{" and tran[-1] == "}":
        tran = tran[1:-1]
    return tran.lower()


def compute_accuracy(sentences, predicted_tags):
    correct = 0
    almost_correct = 0
    correct_without_segmentation = 0
    total = 0

    for key in sentences:
        for i in range(len(sentences[key])):
            total += 1
            c = sentences[key][i][1] + sentences[key][i][2] if not sentences[key][i][2] is None else sentences[key][i][1]
            pred = predicted_tags[key][i]
            if is_equal(c, pred):
                correct += 1
            elif is_equal(to_canonical_rep(c), to_canonical_rep(pred)):
                almost_correct += 1
            elif is_equal_without_segmentation(c, pred):
                correct_without_segmentation += 1
            else:
                print(c)
                print(pred)
                print("#####")

    print(total)

    if total != 0:
        return correct / float(total), (correct + almost_correct) / float(total), \
               (correct + correct_without_segmentation) / float(total)
    else:
        return "None", "None", "None"


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
    all_predicted_tags = {}
    for key in formated_sentences:
        line_predicted_tags = prediction_function(formated_sentences[key], *args)
        predicted_tags[key] = line_predicted_tags
        if algorithm == "biLSTM":
            predicted_tags[key] = line_predicted_tags[0]
            all_predicted_tags[key] = line_predicted_tags

    #print(algorithm + "predictions: " + str(predicted_tags))
    accuracy, almost_accuracy, accuracy_without_segmentation = compute_accuracy(sentences, predicted_tags)
    print(algorithm + " percentage: " + str(accuracy))
    print(algorithm + " percentage regardless of logogram or determivative: " + str(almost_accuracy))
    print(algorithm + " percentage regardless of segmentation: " + str(accuracy_without_segmentation))
    if algorithm == "biLSTM":
        return accuracy, almost_accuracy, accuracy_without_segmentation, all_predicted_tags
    return accuracy, almost_accuracy, accuracy_without_segmentation, predicted_tags


def biLSTM_predict(line, id_to_tran, predictor_from_file, model_from_file):
    tag_logits = predictor_from_file.predict(line)['tag_logits']
    return logits_to_trans(tag_logits, model_from_file, id_to_tran)


def make_combined_prediction(sentences, HMM_predicted_tags, MEMM_predicted_tags, biLSTM_predicted_tags_and_scores):
    # values decided to be best by check on rinap corpus
    #gamma1 = 0.4
    #gamma2 = 0.2

    gamma1 = 1.5
    gamma2 = 1.5

    combined_tags = {}
    for key in HMM_predicted_tags:
        algorithms_tag = biLSTM_predicted_tags_and_scores[key] + (HMM_predicted_tags[key], MEMM_predicted_tags[key])
        line_predicted_tags = combine_tags(algorithms_tag, gamma1, gamma2)
        #line_predicted_tags = prediction_function(formated_sentences[key], *args)
        combined_tags[key] = line_predicted_tags

    accuracy, almost_accuracy, accuracy_without_segmentation = compute_accuracy(sentences, combined_tags)
    print("combined percentage: " + str(accuracy))
    print("combined percentage regardless of logogram or determivative: " + str(almost_accuracy))
    print("combined percentage regardless of segmentation: " + str(accuracy_without_segmentation))

    return accuracy, almost_accuracy, accuracy_without_segmentation


def make_predictions(sentences, lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, sign_to_id,
                        id_to_tran, predictor_from_file, model_from_file):
    HMM_accuracy_and_predictions = make_algorithm_prediction("HMM", sentences, parsed_json_to_HMM_format,
        sign_to_id, hmm_viterbi, 0, {}, {}, {}, {}, {}, lambda1, lambda2)

    MEMM_accuracy_and_predictions = make_algorithm_prediction("MEMM", sentences, parsed_json_to_HMM_format,
        sign_to_id, memm_greedy, logreg, vec, idx_to_tag_dict, extra_decoding_arguments)

    biLSTM_accuracy_and_predictions = make_algorithm_prediction("biLSTM", sentences,
        parsed_json_to_allen_format, sign_to_id,  biLSTM_predict, id_to_tran, predictor_from_file, model_from_file)

    combined_accuracy = make_combined_prediction(sentences, HMM_accuracy_and_predictions[-1],
                                                 MEMM_accuracy_and_predictions[-1], biLSTM_accuracy_and_predictions[-1])

    return HMM_accuracy_and_predictions, MEMM_accuracy_and_predictions, biLSTM_accuracy_and_predictions, \
           combined_accuracy


def operate_on_file(directory, corpus, file, lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments,
                    sign_to_id, id_to_tran, predictor_from_file, model_from_file):
    print(file)
    f = directory / corpus / file
    parsed, _, _ = parse_json(f)

    dict = {}
    dict[file] = parsed
    sentences = break_into_sentences(dict)
    #print(sentences)
    HMM_accuracy, MEMM_accuracy, biLSTM_accuracy, combined_accuracy = make_predictions(sentences, lambda1, lambda2,
        logreg, vec, idx_to_tag_dict, extra_decoding_arguments, sign_to_id, id_to_tran,
        predictor_from_file, model_from_file)

    dialect = get_dialect(corpus, file)
    print(dialect)

    if biLSTM_accuracy[0] == 'None':
        return

    global dialects_HMM
    global dialects_MEMM
    global dialects_biLSTM
    global dialects_combined
    if dialect not in dialects_HMM:
        dialects_HMM[dialect] = [(HMM_accuracy[0], HMM_accuracy[1], HMM_accuracy[2])]
        dialects_MEMM[dialect] = [(MEMM_accuracy[0], MEMM_accuracy[1], MEMM_accuracy[2])]
        dialects_biLSTM[dialect] = [(biLSTM_accuracy[0], biLSTM_accuracy[1], biLSTM_accuracy[2])]
        dialects_combined[dialect] = [(combined_accuracy[0], combined_accuracy[1], combined_accuracy[2])]
    else:
        dialects_HMM[dialect].append((HMM_accuracy[0], HMM_accuracy[1], HMM_accuracy[2]))
        dialects_MEMM[dialect].append((MEMM_accuracy[0], MEMM_accuracy[1], MEMM_accuracy[2]))
        dialects_biLSTM[dialect].append((biLSTM_accuracy[0], biLSTM_accuracy[1], biLSTM_accuracy[2]))
        dialects_combined[dialect].append((combined_accuracy[0], combined_accuracy[1], combined_accuracy[2]))

    print(dialects_HMM)
    print(dialects_MEMM)
    print(dialects_biLSTM)
    print(dialects_combined)


def one_dict_to_three(dict):
    d = {}
    almost_d = {}
    d_without_segmentation = {}

    for dialect in dict:
        d[dialect] = []
        almost_d[dialect] = []
        d_without_segmentation[dialect] = []
        for l in dict[dialect]:
            d[dialect].append(l[0])
            almost_d[dialect].append(l[1])
            d_without_segmentation[dialect].append(l[2])

    return d, almost_d, d_without_segmentation


def print_algorithm_averages(alg, dict):
    regular, almost, without_sementation = one_dict_to_three(dict)

    for dialect in dict:
        print(dialect + " " + alg + " average: " + str(statistics.mean(regular[dialect])) + " of " +
              str(len(regular[dialect])) + " texts.")
        print(dialect + " " + alg + " regardless of case and parenthesis average: " +
              str(statistics.mean(almost[dialect])) + " of " + str(len(almost[dialect])) + " texts.")
        print(dialect + " " + alg + " regardless of segmentation average: " +
              str(statistics.mean(without_sementation[dialect])) + " of " + str(len(without_sementation[dialect])) +
              " texts.")


def compute_averages():
    print_algorithm_averages("HMM", dialects_HMM)
    print_algorithm_averages("MEMM", dialects_MEMM)
    print_algorithm_averages("biLSTM", dialects_biLSTM)
    print_algorithm_averages("combined", dialects_combined)


def main():
    directory = Path(r"../raw_data/test_texts")
    global dialects_HMM
    dialects_HMM = {}
    global dialects_MEMM
    dialects_MEMM = {}
    global dialects_biLSTM
    dialects_biLSTM = {}
    global dialects_combined
    dialects_combined = {}

    lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, sign_to_id, id_to_tran, \
    predictor_from_file, model_from_file = copied_code_from_translate_Akkadian()

    #corpus = "random"
    #for file in os.listdir(directory / corpus):
    #    operate_on_file(directory, corpus, file, lambda1, lambda2, logreg, vec, idx_to_tag_dict,
    #                    extra_decoding_arguments, sign_to_id, id_to_tran, predictor_from_file, model_from_file)

    corpus = "riao"
    for file in os.listdir(directory / corpus):
        operate_on_file(directory, corpus, file, lambda1, lambda2, logreg, vec, idx_to_tag_dict,
                        extra_decoding_arguments, sign_to_id, id_to_tran, predictor_from_file, model_from_file)

    compute_averages()

'''
    inner_directory = directory / "ribo"
    corpus_base = "babylon"
    for i in [2, 3, 4, 5, 6, 7, 8, 10]:
        corpus = corpus_base + str(i)
        for file in os.listdir(inner_directory / corpus):
            operate_on_file(inner_directory, corpus, file, lambda1, lambda2, logreg, vec, idx_to_tag_dict,
                            extra_decoding_arguments, sign_to_id, id_to_tran, predictor_from_file, model_from_file)
'''

if __name__ == '__main__':
    main()
