from pathlib import Path
from data import dump_object_to_file, load_object_from_file, compute_accuracy
from build_data import preprocess
from hmm import hmm_train, hmm_preprocess, hmm_viterbi
from memm import memm_train, build_extra_decoding_arguments, memm_greedy
from BiLSTM import prepare1, prepare2, BiLSTM_predict
from allennlp.predictors import SentenceTaggerPredictor


def hmm_train_and_store():
    train_texts, dev_texts, test_texts, sign_to_id, tran_to_id, id_to_sign, id_to_tran = preprocess()
    most_common_tag, possible_tags, q, e, S, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2 = \
        hmm_train(train_texts, dev_texts)

    dump_object_to_file((most_common_tag, possible_tags, q, e, S, total_tokens, q_bi_counts, q_uni_counts, lambda1,
                         lambda2, sign_to_id, tran_to_id, id_to_sign, id_to_tran), r"..\output\hmm_model")

    return train_texts, dev_texts, test_texts


def memm_and_hmm_train_and_store():
    train_texts, dev_texts, test_texts = hmm_train_and_store()

    memm_train(train_texts, dev_texts)
    logreg, vec, idx_to_tag_dict = memm_train(train_texts, dev_texts)
    extra_decoding_arguments = build_extra_decoding_arguments(train_texts)

    dump_object_to_file((logreg, vec, idx_to_tag_dict, extra_decoding_arguments), r"..\output\memm_model")

    return test_texts


def biLSTM_memm_and_hmm_train_and_store():
    test_texts = memm_and_hmm_train_and_store()

    model, vocab, train_dataset, validation_dataset, cuda_device, reader = prepare1()
    trainer, model, reader, vocab = prepare2(model, vocab, train_dataset, validation_dataset, cuda_device, reader)
    trainer.train()
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

    dump_object_to_file((model, predictor), r"..\output\biLSTM_model")

    return test_texts


def load_trained_data():
    most_common_tag, possible_tags, q, e, S, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2, sign_to_id, \
    tran_to_id, id_to_sign, id_to_tran = load_object_from_file(Path("../output/hmm_model.pkl"))

    logreg, vec, idx_to_tag_dict, extra_decoding_arguments = load_object_from_file(Path("../output/memm_model.pkl"))

    model, predictor = load_object_from_file(Path("../output/biLSTM_model.pkl"))

    return most_common_tag, possible_tags, q, e, S, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2, \
           sign_to_id, tran_to_id, id_to_sign, id_to_tran, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, \
           model, predictor


def test_trained_data(test_texts, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2, logreg, vec,
                      idx_to_tag_dict, model, predictor, sign_to_id, id_to_tran):
    print(compute_accuracy(test_texts, hmm_viterbi, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2))
    print(compute_accuracy(test_texts, memm_greedy, logreg, vec, idx_to_tag_dict))
    print(compute_accuracy(test_texts, BiLSTM_predict, model, predictor, sign_to_id, id_to_tran))


def main():
    test_texts = biLSTM_memm_and_hmm_train_and_store()
    most_common_tag, possible_tags, q, e, S, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2, \
    sign_to_id, tran_to_id, id_to_sign, id_to_tran, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, \
    model, predictor = load_trained_data()

    test_trained_data(test_texts, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2, logreg, vec,
                      idx_to_tag_dict, model, predictor, sign_to_id, id_to_tran)


if __name__ == '__main__':
    main()
