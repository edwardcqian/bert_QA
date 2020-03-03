import numpy as np
import torch

from transformers import (
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    CamembertConfig,
    CamembertForQuestionAnswering,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
)


MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "camembert": (
        CamembertConfig, CamembertForQuestionAnswering, CamembertTokenizer
    ),
    "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    "distilbert": (
        DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer
    ),
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
}


class ModelInference:
    def __init__(
        self, model_dir, model_type='bert', do_lower=True, use_gpu=False
    ):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        )
        config_class, model_class, tokenizer_class = MODEL_CLASSES[
            model_type
        ]
        self.config = config_class.from_pretrained(model_dir)
        self.tokenizer = tokenizer_class.from_pretrained(
            model_dir, do_lower_case=do_lower,
        )
        checkpoint = model_dir
        self.model = model_class.from_pretrained(checkpoint)

        self.model.to(self.device)

    def add_target_text(self, data):
        self.target_text = self.tokenizer.tokenize(data)

    def prep_question(self, data):
        question_tokens = self.tokenizer.tokenize(data)
        input_ids = (
            ['[CLS]'] + question_tokens +
            ['[SEP]'] + self.target_text + ['[SEP]']
        )
        attention_mask = np.zeros(self.tokenizer.max_len)
        attention_mask[:(len(input_ids)-1)] = 1
        token_type_ids = np.zeros(self.tokenizer.max_len)
        token_type_ids[:(len(input_ids)-1)] = 1
        token_type_ids[:(len(question_tokens)+1)] = 0
        input_ids = np.array(self.tokenizer.encode(
            input_ids, pad_to_max_length=True,
        ))
        input_ids = torch.from_numpy(
            input_ids.reshape(1, self.tokenizer.max_len)
        )
        attention_mask = torch.from_numpy(
            attention_mask.reshape(1, self.tokenizer.max_len)
        )
        token_type_ids = torch.from_numpy(
            token_type_ids.reshape(1, self.tokenizer.max_len)
        )

        return [input_ids, attention_mask, token_type_ids]

    def evaluate(self, data):
        pred_data = self.prep_question(data)
        pred_data = tuple(t.to(self.device).to(torch.int64) for t in pred_data)
        self.model.eval()

        with torch.no_grad():
            inputs = {
                "input_ids": pred_data[0],
                "attention_mask": pred_data[1],
                "token_type_ids": pred_data[2],
            }
            outputs = self.model(**inputs)
            start_logits = outputs[0].detach().cpu().tolist()
            end_logits = outputs[1].detach().cpu().tolist()
            start_idx = np.argmax(start_logits)
            end_idx = np.argmax(end_logits)
            input_data = self.tokenizer.convert_ids_to_tokens(
                pred_data[0].detach().cpu().tolist()[0]
            )
        answer = 'No valid answer found'
        if start_idx < end_idx:
            answer = self.tokenizer.convert_tokens_to_string(
                input_data[start_idx:(end_idx+1)]
            )
        return answer
