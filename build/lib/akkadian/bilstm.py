from typing import Iterator, List, Dict
import torch
import torch.optim as optim
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
import platform
from pathlib import Path
from akkadian.build_data import preprocess
from akkadian.data import dump_object_to_file, load_object_from_file, logits_to_trans, compute_accuracy


torch.manual_seed(1)
class PosDatasetReader(DatasetReader):
    """
    class based on AllenNLP tutorial (https://allennlp.org/tutorials)
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                pairs = line.strip().split()
                sentence, tags = zip(*(pair.split("###") for pair in pairs))
                yield self.text_to_instance([Token(word) for word in sentence], tags)


class LstmTagger(Model):
    """
    class based on AllenNLP tutorial (https://allennlp.org/tutorials)
    """
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


def BiLSTM_predict(text, model, predictor, sign_to_id, id_to_tran):
    """
    Predict the transliteration of a sentence of signs using biLSTM
    :param text: sentence to predict
    :param model: biLSTM model object
    :param predictor: biLSTM predictor object
    :param sign_to_id: dictionary mapping signs to ids
    :param id_to_tran: dictionary mapping ids to transliterations
    :return: transliteration prediction for text
    """
    allen_format = ""
    for sign, tran in text:
        allen_format += str(sign_to_id[sign]) + " "
    allen_format = allen_format[:-1]

    tag_logits = predictor.predict(allen_format)['tag_logits']
    prediction, _, _, _, _, _ = logits_to_trans(tag_logits, model, id_to_tran)
    return prediction


def prepare1():
    """
    First part of preparing data for training
    :return: biLSTM model object, biLSTM vocabulary, data for training, data for validation, cuda biLSTM object,
             biLSTM reader object
    """
    reader = PosDatasetReader()
    if platform.system() == "Windows":
        train_dataset = reader.read(r"..\BiLSTM_input\allen_train_texts.txt")
        validation_dataset = reader.read(r"..\BiLSTM_input\allen_dev_texts.txt")
    else:
        train_dataset = reader.read(r"../BiLSTM_input/allen_train_texts.txt")
        validation_dataset = reader.read(r"../BiLSTM_input/allen_dev_texts.txt")

    vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

    EMBEDDING_DIM = 200
    HIDDEN_DIM = 200

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True))

    model = LstmTagger(word_embeddings, lstm, vocab)
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    return model, vocab, train_dataset, validation_dataset, cuda_device, reader


def prepare2(model, vocab, train_dataset, validation_dataset, cuda_device, reader):
    """
    Second part of preparing data for training
    :param model: biLSTM model object
    :param vocab: biLSTM vocabulary
    :param train_dataset: data for training
    :param validation_dataset: data for validation
    :param cuda_device: cuda biLSTM object
    :param reader: biLSTM reader object
    :return: trainer biLSRM obejct, biLSTM model obkect, biLSTM reader object and biLSTM vocabulary
    """
    optimizer = optim.SGD(model.parameters(), lr=0.3)
    iterator = BucketIterator(batch_size=1, sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      #patience=1,
                      patience=10,
                      #num_epochs=2,
                      num_epochs=1000,
                      cuda_device=cuda_device)

    return trainer, model, reader, vocab


def train(trainer, model, reader):
    """
    Use trainer object to train the biLSTM model
    :param trainer: trainer object of the biLSTM model
    :param model: biLSTM model object
    :param reader: reader for the biLSTM
    :return: nothing
    """
    trainer.train()
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

    dump_object_to_file(predictor, Path(r".output/predictor"))
    dump_object_to_file(model, Path(r".output/model"))


def check_results(train_texts, dev_texts, test_texts, sign_to_id, id_to_tran):
    """
    Prints the accuracy of the trained biLSTM models
    :param train_texts: texts used for train
    :param dev_texts: texts used for dev
    :param test_texts: texts used for test
    :param sign_to_id: dictionary mapping signs to ids
    :param id_to_tran: dictionary mapping ids to transliterations
    :return: nothing
    """
    predictor_from_file = load_object_from_file(Path(r".output/predictor"))
    model_from_file = load_object_from_file(Path(r".output/model"))

    print(compute_accuracy(train_texts, BiLSTM_predict, model_from_file, predictor_from_file, sign_to_id, id_to_tran))
    print(compute_accuracy(dev_texts, BiLSTM_predict, model_from_file, predictor_from_file, sign_to_id, id_to_tran))
    print(compute_accuracy(test_texts, BiLSTM_predict, model_from_file, predictor_from_file, sign_to_id, id_to_tran))


def main():
    """
    Check the biLSTM model
    :return: nothing
    """
    train_texts, dev_texts, test_texts, sign_to_id, tran_to_id, id_to_sign, id_to_tran = preprocess()
    model, vocab, train_dataset, validation_dataset, cuda_device, reader = prepare1()
    trainer, model, reader, vocab = prepare2(model, vocab, train_dataset, validation_dataset, cuda_device, reader)
    train(trainer, model, reader)
    check_results(train_texts, dev_texts, test_texts, sign_to_id, id_to_tran)


if __name__ == '__main__':
    main()
