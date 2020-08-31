from allennlp.predictors import SentenceTaggerPredictor
from akkadian.data import dump_object_to_file, load_object_from_file, compute_accuracy
from akkadian.build_data import preprocess
from akkadian.hmm import hmm_train, hmm_preprocess, hmm_viterbi
from akkadian.memm import memm_train, build_extra_decoding_arguments, memm_greedy
from akkadian.bilstm import prepare1, prepare2, BiLSTM_predict
from akkadian.__init__ import hmm_path, memm_path, bilstm_path


def hmm_train_and_store(corpora):
    """
    Trains HMM model and stores all the data needed for using HMM
    :return: nothing, stores everything in hmm_model.pkl
    """
    train_texts, dev_texts, test_texts, _, _, _, _ = preprocess(corpora)
    most_common_tag, possible_tags, q, e, S, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2 = \
        hmm_train(train_texts, dev_texts)

    dump_object_to_file((most_common_tag, possible_tags, q, e, S, total_tokens, q_bi_counts, q_uni_counts, lambda1,
                         lambda2, test_texts), hmm_path)


def hmm_train_and_test(corpora):
    """
    Trains HMM model, stores all data and print the accuracy
    :return: nothing, stores everything in hmm_model.pkl
    """
    hmm_train_and_store(corpora)

    most_common_tag, possible_tags, q, e, S, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2, test_texts = \
        load_object_from_file(hmm_path)
    print(compute_accuracy(test_texts, hmm_viterbi, total_tokens, q_bi_counts, q_uni_counts, q, e, S, most_common_tag,
                           possible_tags, lambda1, lambda2))


def memm_train_and_store(corpora):
    """
    Trains MEMM model and stores all the data needed for using MEMM
    :return: nothing, stores everything in memm_model.pkl
    """
    train_texts, dev_texts, test_texts, _, _, _, _ = preprocess(corpora)

    logreg, vec, idx_to_tag_dict = memm_train(train_texts, dev_texts)

    dump_object_to_file((logreg, vec, idx_to_tag_dict, test_texts), memm_path)


def memm_train_and_test(corpora):
    """
    Trains MEMM model, stores all data and print the accuracy
    :return: nothing, stores everything in memm_model.pkl
    """
    memm_train_and_store(corpora)

    logreg, vec, idx_to_tag_dict, test_texts = load_object_from_file(memm_path)
    print(compute_accuracy(test_texts, memm_greedy, logreg, vec, idx_to_tag_dict))


def biLSTM_train_and_store(corpora):
    """
    Trains biLSTM model and stores all the data needed for using biLSTM
    :return: nothing, stores everything in bilstm_model.pkl
    """
    train_texts, dev_texts, test_texts, sign_to_id, _, _, id_to_tran = preprocess(corpora)

    model, vocab, train_dataset, validation_dataset, cuda_device, reader = prepare1()
    trainer, model, reader, vocab = prepare2(model, vocab, train_dataset, validation_dataset, cuda_device, reader)
    trainer.train()
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

    dump_object_to_file((model, predictor, sign_to_id, id_to_tran, test_texts), bilstm_path)


def biLSTM_train_and_test(corpora):
    """
    Trains biLSTM model, stores all data and print the accuracy
    :return: nothing, stores everything in bilstm_model.pkl
    """
    biLSTM_train_and_store(corpora)

    model, predictor, sign_to_id, id_to_tran, test_texts = load_object_from_file(bilstm_path)
    print(compute_accuracy(test_texts, BiLSTM_predict, model, predictor, sign_to_id, id_to_tran))


def main():
    """
    Trains biLSTM, MEMM and HMM models, stores all data and print the accuracies
    :return: nothing, stores everything in pickles
    """

    corpora = ['rinap/rinap1', 'rinap/rinap3', 'rinap/rinap4', 'rinap/rinap5']

    #print('##### HMM #####')
    #hmm_train_and_test(corpora)

    #print('##### MEMM #####')
    #memm_train_and_test(corpora)

    #print('##### BiLSTM #####')
    #biLSTM_train_and_test(corpora)

if __name__ == '__main__':
    main()
