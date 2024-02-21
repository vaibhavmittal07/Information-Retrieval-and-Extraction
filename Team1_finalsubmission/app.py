import numpy as np
import pandas as pd
# import gensim
# import Reference
import os
import csv
import nltk
import gzip
import logging
import sys
import numpy as np
import torch.nn as nn
# import openpyxl
import string
from nltk.tokenize import word_tokenize, sent_tokenize
# nltk.download('punkt')

from bs4 import BeautifulSoup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import tensor
# pip install language_tool_python
import language_tool_python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

tool = language_tool_python.LanguageTool('en-US')

import torch
import numpy as np
from nltk.tokenize import sent_tokenize  # We'll use NLTK to tokenize the essay into sentences
# pip install transformers
from transformers import BertTokenizer, BertModel, AdamW  # Ensure you've imported these
import random

from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image  # Make sure to install the Pillow library: pip install Pillow
import torch.nn as nn
from flask import jsonify
# from flask import Flask, render_template, request, jsonify


app = Flask(__name__)



string.punctuation
punct = string.punctuation
punct = punct.replace('.', '')
punct += '@'
'.' in punct

def get_lower(text):
    return text.lower()

def remove_punctuations(text):
    return ''.join([char for char in text if char not in punct])

def tokenize(text):
    # text = text.strip()
    return word_tokenize(text)

def remove_alpha_numeric(sentence):
    # return ' '.join(word for word in tokens if word.isalpha())
    words = sentence.split()
    alphabetic_words = [word for word in words if word.isalpha()]
    return ' '.join(alphabetic_words)

def remove_tags(text):
    return BeautifulSoup(text, 'html.parser').get_text()

def remove_extra_gaps(text):
    return ' '.join(text.split())

def pipeline(text):
    text = get_lower(text)
    text = remove_punctuations(text)
    # tokens = tokenize(text)
    # text = remove_alpha_numeric(text)
    # text = remove_tags(text)
    text = remove_extra_gaps(text)
    return text


class BertSentenceEmbedding(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super(BertSentenceEmbedding, self).__init__()  # Call the super class's __init__ first

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.model = BertModel.from_pretrained(pretrained_model_name, output_hidden_states=True)
        self.model.eval()

    def get_embedding(self, text):

        # Tokenize the essay into sentences
        sentences = sent_tokenize(text)

        # List to hold embeddings for each sentence
        sentence_embeddings = []

        for sentence in sentences:
            # I noticed you were using a "pipeline" function that was not defined in the given code
            processed_sentence = pipeline(sentence)
            inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use the penultimate layer's hidden states
            hidden_states = outputs.hidden_states[-2]

            # Compute the mean of all tokens embeddings for this sentence
            sentence_embedding = torch.mean(hidden_states, dim=1).squeeze().cpu().numpy()
            sentence_embeddings.append(sentence_embedding)

        return np.array(sentence_embeddings)  # Return embeddings for all sentences in the essay

# Assuming you've defined device somewhere above
bert_embedder = BertSentenceEmbedding()


def preprocess_essay(essays):
    # print(essays)
    essay_embeddings = [torch.tensor(bert_embedder.get_embedding(essay)) for essay in essays]

    # Find the maximum number of sentences in all essays
    max_sentences = max(embed.shape[0] for embed in essay_embeddings)

    # Calculate the number of dimensions (features) in the embeddings
    num_features = essay_embeddings[0].shape[1]  # Assumes all embeddings have the same number of features

    # Pad the sentence embeddings to have the same number of sentences
    padded_embeddings = []
    for embed in essay_embeddings:
        padding = max_sentences - embed.shape[0]
        padded_embed = torch.cat((embed, torch.zeros(padding, num_features)), dim=0)
        padded_embeddings.append(padded_embed)

    embeddings_batch = torch.stack(padded_embeddings)  # batch * sentences * num_features
    lengths_batch = torch.tensor([max_sentences] * len(essays), dtype=torch.int64)

    return embeddings_batch, lengths_batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SemanticScore(nn.Module):
    def __init__(self):
        super(SemanticScore, self).__init__()

        self.bert_emb_dim = 768
        self.dropout_prob = 0.5
        self.lstm_hidden_size = 1024
        self.lstm_layers_num = 1
        self.fnn_hidden_size = []
        self.bidirectional = False

        self.lstm = nn.LSTM(self.bert_emb_dim,
                            self.lstm_hidden_size,
                            self.lstm_layers_num,
                            bidirectional=self.bidirectional,
                            batch_first=True)
        self.dropout = nn.Dropout(self.dropout_prob)

        in_features = self.lstm_hidden_size * 2 if self.bidirectional else self.lstm_hidden_size
        layers = []
        for hs in self.fnn_hidden_size:
            layers.append(nn.Linear(in_features, hs))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_prob))
            in_features = hs

        layers.append(nn.Linear(in_features, 400))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_prob))
        layers.append(nn.Linear(400, 1))
        layers.append(nn.Sigmoid())
        self.fnn = nn.Sequential(*layers)

    def forward(self, batch_doc_encodes, batch_doc_sent_nums):
        packed_input = pack_padded_sequence(batch_doc_encodes, batch_doc_sent_nums, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        logits = self.fnn(output[:, -1, :]) # Using the output of the last timestep
        return logits.squeeze(-1)
    
model_semantic=SemanticScore().to(device)
model_semantic.load_state_dict(torch.load(r"semantic_model_final.pth",map_location=torch.device('cpu')))



class CoherenceScore(nn.Module):
    def __init__(self):
        super(CoherenceScore, self).__init__()

        self.bert_emb_dim = 768
        self.dropout_prob = 0.5
        self.lstm_hidden_size = 1024
        self.lstm_layers_num = 2
        self.fnn_hidden_size = []
        self.bidirectional = False

        self.lstm = nn.LSTM(self.bert_emb_dim,
                            self.lstm_hidden_size,
                            self.lstm_layers_num,
                            bidirectional=self.bidirectional,
                            batch_first=True)
        self.dropout = nn.Dropout(self.dropout_prob)

        in_features = self.lstm_hidden_size * 2 if self.bidirectional else self.lstm_hidden_size
        layers = []
        # for hs in self.fnn_hidden_size:
        #     layers.append(nn.Linear(in_features, hs))
        #     layers.append(nn.ReLU())
        #     layers.append(nn.Dropout(self.dropout_prob))
        #     in_features = hs

        layers.append(nn.Linear(in_features, 400))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_prob))
        layers.append(nn.Linear(400, 1))
        layers.append(nn.Sigmoid())
        self.fnn = nn.Sequential(*layers)

    def forward(self, batch_doc_encodes, batch_doc_sent_nums):
        packed_input = pack_padded_sequence(batch_doc_encodes, batch_doc_sent_nums, batch_first=True, enforce_sorted=False)
        # print(packed_input.data.shape,"packed_input")
        packed_output, _ = self.lstm(packed_input)
        # print(packed_output.data.shape,"packed_out")
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        logits = self.fnn(output[:, -1, :]) # Using the output of the last timestep
        return logits.squeeze(-1)

model_coher=CoherenceScore().to(device)
model_coher.load_state_dict(torch.load(r"coherence_model_final.pth",map_location=torch.device('cpu')))


class PromptEmbedding(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super(PromptEmbedding, self).__init__()  # Call the super class's __init_ first

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.model = BertModel.from_pretrained(pretrained_model_name, output_hidden_states=True)
        self.model.eval()


    def get_embedding(self, prompt, essay):
        essay = sent_tokenize(essay)
        sentence_embeddings = []

        for sentence in essay:
            processed_sentence = pipeline(sentence)
            inputs = self.tokenizer(prompt, sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs.to(self.device))

            hidden_states = outputs.hidden_states[-4:]
            # print("hidden_states :",hidden_states[0].shape)

            # Concatenate the hidden states along the third dimension
            concatenated_hidden_states = torch.cat(hidden_states, dim=1)
            # print("concatenated_hidden_states :",concatenated_hidden_states.shape)

            # Compute the mean of all tokens embeddings for this sentence along the second dimension
            sentence_embedding = torch.mean(concatenated_hidden_states, dim=1).squeeze().cpu().numpy()
            # print("sentence_embedding :",sentence_embedding.shape)

            sentence_embeddings.append(sentence_embedding)

        return np.array(sentence_embeddings)

prompt_embedder = PromptEmbedding().to(device)


def prompt_essay(prompts, essays):
    # print(essays)
    essay_embeddings = [torch.tensor(prompt_embedder.get_embedding(prompt, essay)) for prompt, essay in zip(prompts, essays)]

    # Find the maximum number of sentences in all essays
    max_sentences = max(embed.shape[0] for embed in essay_embeddings)

    # Calculate the number of dimensions (features) in the embeddings
    num_features = essay_embeddings[0].shape[1]  # Assumes all embeddings have the same number of features

    # Pad the sentence embeddings to have the same number of sentences
    padded_embeddings = []
    for embed in essay_embeddings:
        padding = max_sentences - embed.shape[0]
        padded_embed = torch.cat((embed, torch.zeros(padding, num_features)), dim=0)
        padded_embeddings.append(padded_embed)

    embeddings_batch = torch.stack(padded_embeddings)  # batch * sentences * num_features
    lengths_batch = torch.tensor([max_sentences] * len(essays), dtype=torch.int64)

    return embeddings_batch, lengths_batch


class PromptScore(nn.Module):
    def __init__(self):
        super(PromptScore, self).__init__()
        self.bert_emb_dim = 768
        self.dropout_prob = 0.5
        self.lstm_hidden_size = 1024
        self.lstm_layers_num = 1
        self.fnn_hidden_size = []
        self.bidirectional = False

        self.lstm = nn.LSTM(self.bert_emb_dim,
                            self.lstm_hidden_size,
                            self.lstm_layers_num,
                            bidirectional=self.bidirectional,
                            batch_first=True)
        self.dropout = nn.Dropout(self.dropout_prob)

        in_features = self.lstm_hidden_size * 2 if self.bidirectional else self.lstm_hidden_size
        layers = []
        for hs in self.fnn_hidden_size:
            layers.append(nn.Linear(in_features, hs))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_prob))
            in_features = hs

        layers.append(nn.Linear(in_features, 400))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_prob))
        layers.append(nn.Linear(400, 1))
        layers.append(nn.Sigmoid())
        self.fnn = nn.Sequential(*layers)

    def forward(self, batch_doc_encodes, batch_doc_sent_nums):
        packed_input = pack_padded_sequence(batch_doc_encodes, batch_doc_sent_nums, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        logits = self.fnn(output[:, -1, :]) # Using the output of the last timestep
        return logits.squeeze(-1)

model_prompt=PromptScore().to(device)
model_prompt.load_state_dict(torch.load(r"prompt_model_final.pth",map_location=torch.device('cpu')))

def get_grammatical_score(text, tool=tool):
    size = len(text.split())
    num = len(tool.check(text))
    # print(size, num)
    return (size - num) / size

def get_word_count(text):
    return len(text.split())

def get_char_count(text):
    return len(text)

def get_mean_score(text):
    words = text.split()
    word_lengths = [len(word) for word in words]
    mean_word_length = sum(word_lengths) / len(word_lengths)
    return mean_word_length

def get_variance_score(text):
    words = text.split()
    word_lengths = [len(word) for word in words]
    mean_word_length = sum(word_lengths) / len(word_lengths)
    variance_word_length = sum((length - mean_word_length) ** 2 for length in word_lengths) / len(word_lengths)
    return variance_word_length


class CustomLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, predicted_scores, labels):
        # Convert predicted_scores and labels to 1D tensors
        predicted_scores = predicted_scores.view(-1)
        labels = labels.view(-1)

        # SIM loss
        sim_loss = 1 - F.cosine_similarity(predicted_scores, labels, dim=0)

        # MR loss
        mr_loss = F.mse_loss(predicted_scores, labels)

        # Coherence loss
        coherence_loss = torch.mean(torch.abs(predicted_scores - labels))

        # Combine losses
        loss = self.alpha * sim_loss + self.beta * mr_loss + self.gamma * coherence_loss

        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class SegmentScaleEssayModelo(nn.Module):
    def __init__(self, bert_model, lstm_hidden_size, segment_scales):
        super(SegmentScaleEssayModelo, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.lstm = nn.LSTM(input_size=768, hidden_size=lstm_hidden_size, batch_first=True, dropout=0.1)
        self.attention_pooling = nn.Linear(lstm_hidden_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.segment_scales = segment_scales
        self.lstm_hidden=lstm_hidden_size

        for param in self.bert.parameters():
             param.requires_grad = False
        # Create dense regression layers for each segment-scale with dropout
        self.regression_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(lstm_hidden_size, 512),  # Adjust the size as needed
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        ) for _ in segment_scales])

    def forward(self, input_ids, attention_mask):

        batch_size, num_segments, max_tokens = input_ids.size()

        # Reshape input_ids and attention_mask for BERT processing
        input_ids_flat = input_ids.view(batch_size * num_segments, max_tokens)
        attention_mask_flat = attention_mask.view(batch_size * num_segments, max_tokens)

        # Step 1: BERT Processing
        outputs = self.bert(input_ids_flat, attention_mask=attention_mask_flat)
        sequence_outputs = outputs.last_hidden_state

        # Reshape sequence_outputs back to 3D
        sequence_outputs = sequence_outputs.view(batch_size, num_segments, max_tokens, -1)

        # Initialize a list to store segment outputs
        segment_outputs = []

        for segment_index in range(num_segments):
            # Select the current segment from the 3D tensor
            current_segment = sequence_outputs[:, segment_index, :, :]

            # Step 2: LSTM Processing for the current segment
            lstm_outputs, _ = self.lstm(current_segment)

            # Step 3: Attention Pooling for the current segment
            attention_scores = self.tanh(self.attention_pooling(lstm_outputs))
            attention_weights = self.softmax(attention_scores)
            segment_scale_representation = torch.sum(attention_weights * lstm_outputs, dim=1)

            # Append the segment representation to the list
            segment_outputs.append(segment_scale_representation)

        # Concatenate segment outputs to get the final representation

        final_representation = torch.cat(segment_outputs, dim=1)

        final_representation=final_representation.reshape(batch_size,num_segments,self.lstm_hidden)

        # print(final_representation.shape)

        return final_representation

    def segment_scale_representation(self, input_ids, attention_mask):
        # Forward pass through the model for each segment-scale

        segment_outputs = []
        for scale in self.segment_scales:
            # Calculate the number of segments for each scale
            num_segments = (input_ids.size(1) + scale - 1) // scale

            # Pad the input_ids and attention_mask to fit the segments
            pad_tokens = num_segments * scale - input_ids.size(1)
            input_ids_padded = F.pad(input_ids, (0, pad_tokens), value=tokenizer.pad_token_id)
            attention_mask_padded = F.pad(attention_mask, (0, pad_tokens), value=0)

            # Reshape input_ids and attention_mask into segments
            segment_input_ids = input_ids_padded.view(-1, num_segments, scale)  # Updated this line
            # print(scale)
            # print(segment_input_ids.shape)
            segment_attention_mask = attention_mask_padded.view(-1, num_segments, scale)  # Updated this line


            # Forward pass through the model for each segment-scale
            segment_output = self.forward(segment_input_ids, segment_attention_mask)
            segment_outputs.append(segment_output)

        # Concatenate segment outputs along the sequence dimension
        final_representation = torch.cat(segment_outputs, dim=1)

        # Apply dense regression layer for each segment-scale
        segment_scores = [layer(final_representation) for layer in self.regression_layers]

        # Concatenate segment scores
        final_scores = torch.cat(segment_scores, dim=1)

        # Sum scores across segment-scales to get the final score
        final_score = torch.mean(final_scores, dim=1)


        return final_score


class EssayBERTModel_(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', regression_output_size=1):
        super(EssayBERTModel_, self).__init__()

        # BERT model and tokenizer
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        for param in self.bert.parameters():
            param.requires_grad = False

        # Regression layer with dropout
        self.regression_layer = nn.Sequential(
            nn.Linear(2 * self.bert.config.hidden_size, 512),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(512, regression_output_size)
        )

    def forward(self,essays):
        # Tokenize input essay

        tokenized_input = tokenizer(essays, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

        input_ids = tokenized_input['input_ids']
        attention_mask = tokenized_input['attention_mask']

        batch_size, max_tokens = input_ids.size()

        # Add an extra dimension for num_segments
        input_ids = input_ids.unsqueeze(1)
        attention_mask = attention_mask.unsqueeze(1)

        # Reshape input_ids and attention_mask for BERT processing
        input_ids_flat = input_ids.view(batch_size * 1, max_tokens)
        attention_mask_flat = attention_mask.view(batch_size * 1, max_tokens)

        # Step 1: BERT Processing
        outputs = self.bert(input_ids_flat, attention_mask=attention_mask_flat)

        # Max pooling over the sequence outputs for token-scale representation
        token_representation, _ = torch.max(outputs.last_hidden_state, dim=1)

        # Document-scale representation (pooler output)
        document_representation = outputs.pooler_output

        # Concatenate the two representations
        concatenated_representation = torch.cat([document_representation, token_representation], dim=1)

        # Pass through the regression layer
        output_scores = self.regression_layer(concatenated_representation)

        return output_scores


class CombinedEssayModel_(nn.Module):
    def __init__(self, bert_model, lstm_hidden_size, segment_scales, regression_output_size=1):
        super(CombinedEssayModel_, self).__init__()

        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

        # Segment-scale model
        self.segment_scale_model = SegmentScaleEssayModelo(bert_model, lstm_hidden_size, segment_scales)

        # Document-scale and Token-scale model
        self.essay_bert_model = EssayBERTModel_(bert_model, regression_output_size)

    def forward(self, input_text, max_tokens):

        tokenized_input = tokenizer(input_text, return_tensors='pt', max_length=max_tokens, truncation=True, padding='max_length')

        input_ids = tokenized_input['input_ids']
        attention_mask = tokenized_input['attention_mask']
        # Get segment-scale representation
        segment_scale_representation_score = self.segment_scale_model.segment_scale_representation(input_ids, attention_mask)

        # print(segment_scale_representation_score)
        # Get document-scale and token-scale representation
        essay_representation_score = self.essay_bert_model(input_text)

        # print(essay_representation_score)
        score=essay_representation_score+segment_scale_representation_score

        return score

model_combined = CombinedEssayModel_(bert_model='bert-base-uncased', lstm_hidden_size=256, segment_scales=[5, 10, 25], regression_output_size=1)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
loaded_model_checkpoint = torch.load('entire_model.pt', map_location=torch.device('cpu'))
# print("="*80)
# print(loaded_model_checkpoint.keys())
# print("="*80)
loaded_model_state_dict = loaded_model_checkpoint['model_state_dict']
model_combined.load_state_dict(loaded_model_state_dict, strict=False)

# model_combined.load_state_dict(loaded_model_state_dict)



bert = BertModel.from_pretrained('bert-base-uncased')
# bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

cls = bert_tokenizer.cls_token
sep = bert_tokenizer.sep_token
pad = bert_tokenizer.pad_token
unk = bert_tokenizer.unk_token

cls_id = bert_tokenizer.cls_token_id
sep_id = bert_tokenizer.sep_token_id
pad_id = bert_tokenizer.pad_token_id
unk_id = bert_tokenizer.unk_token_id
# print(cls_id,sep_id,pad_id,unk_id)


#Helper functions for pre-processing
def limit_sentence_length(sentence):
  sentence = sentence.split()
  sentence = sentence[:128]
  return " ".join(sentence)

def get_token_type(sentence,num):
  return [num]*len(sentence)



#Model
class Bert_Model(nn.Module):
  def __init__(self,output_dim):
    super().__init__()
    self.bert = bert
    embedding_dim = bert.config.to_dict()['hidden_size']
    self.out = nn.Linear(embedding_dim, output_dim)
  def forward(self, seq, attention_mask, token_type):
    embeddings = self.bert(input_ids = seq, attention_mask = attention_mask, token_type_ids= token_type)[1]
    return self.out(embeddings)


bert_model_snli = Bert_Model(3).to(device)
bert_model_snli.load_state_dict(torch.load(r"bert_model_snli.pth",map_location=torch.device('cpu')))
optimizer_snli = AdamW(bert_model_snli.parameters(),lr=2e-5,eps=1e-6,correct_bias=False)
criterion_snli = nn.CrossEntropyLoss().to(device)


def predict_inference(premise, hypothesis, bert_model):
    torch.cuda.empty_cache()
    bert_model.eval()
    premise = cls + ' ' + premise + ' ' + sep
    hypothesis = hypothesis + ' ' + sep

    premise_tokens = bert_tokenizer.tokenize(premise)
    hypothesis_tokens = bert_tokenizer.tokenize(hypothesis)

    premise_token_type = get_token_type(premise_tokens,0)
    hypothesis_token_type = get_token_type(hypothesis_tokens,1)
    # print(premise_token_type, hypothesis_token_type)
    seq = premise_tokens + hypothesis_tokens
    seq = bert_tokenizer.convert_tokens_to_ids(seq)
    # print(seq)
    tokens_type = premise_token_type + hypothesis_token_type
    attention_mask = get_token_type(seq,1)

    seq = torch.LongTensor(seq).unsqueeze(0).to(device)
    tokens_type = torch.LongTensor(tokens_type).unsqueeze(0).to(device)
    attention_mask = torch.LongTensor(attention_mask).unsqueeze(0).to(device)

    prediction = bert_model(seq, attention_mask, tokens_type)
    prediction = F.softmax(prediction, dim=-1)

    return prediction[0, 0].item()



def coherence_score_snli(essay):
    paragraphs = essay.split('\n\n')
    final_sum = 0
    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)
        # print(sentences)
        size = len(sentences)-1
        # print(size)
        sum = 0
        for i in range(size):
            x = predict_inference(sentences[i], sentences[i+1], bert_model_snli)
            # print(sentences[i], sentences[i+1], x)
            sum += x
        # print(sum, size, "score :",sum / size)
        final_sum += sum / max(1, size)

    # print("final_sum :", final_sum/len(paragraphs))
    return final_sum/len(paragraphs)

import re
import random as scale

# file type contains {'bert-base-uncased', 'roberta-base', 'xlnet-base-cased'}
file = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(file, sep_token='[SEP]')
url_replacer = '<url>'
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'

MAX_SENTLEN = 50
MAX_SENTNUM = 100

asap_ranges = {
    0: (-60, 60),
    1: (-10, 10),
    2: (-5, 5),
    3: (-3, 3),
    4: (-3, 3),
    5: (-4, 4),
    6: (-4, 4),
    7: (-30, 30),
    8: (-60, 60)
}
def get_ref_dtype():
    return ref_scores_dtype


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens


def get_score_range(prompt_id):
    return asap_ranges[prompt_id]

def scaled_value(low, high):
    return scale.uniform(low, high)

import re
# import nltk

# Download the necessary resource if not already downloaded
# nltk.download('punkt')

def text_tokenizer(text, replace_url_flag=True, tokenize_sent_flag=True, create_vocab_flag=True):
    text = replace_url(text)
    text = text.replace(u'"', u'')

    if "..." in text:
        text = re.sub(r'\.{3,}(\s+\.{3,})*', '...', text)
    if "??" in text:
        text = re.sub(r'\?{2,}(\s+\?{2,})*', '?', text)
    if "!!" in text:
        text = re.sub(r'\!{2,}(\s+\!{2,})*', '!', text)

    # Use nltk word tokenizer
    tokens = nltk.word_tokenize(text)

    if tokenize_sent_flag:
        punctuation = '.!,;:?"\'、，；'
        text = " ".join(tokens)
        text_nopun = re.sub(r'[{}]+'.format(punctuation), '', text)
        sent_tokens = text_nopun


        return sent_tokens
    else:
        raise NotImplementedError

def read_input_essay(input):
    

    data_id = []

    # tokenize text into sentences
    sent_tokens = text_tokenizer(input, replace_url_flag=True, tokenize_sent_flag=True)
    tokenized_text = bert_tokenizer.tokenize(sent_tokens)
    max_num = 512
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)


    data_id.append(indexed_tokens)


    return data_id

import numpy as np
import math

def padding_sentence_sequences_input(index_sequences, maxnum, post_padding=True):


    index_sequences = np.array(index_sequences)
    num_seq = math.ceil((index_sequences.size) / maxnum)
    index_sequences = index_sequences.flatten()

    X = np.empty([num_seq, maxnum], dtype=np.int32)
    mask = np.zeros([num_seq, maxnum], dtype=np.float32)

    j = 0

    for i in range(0, len(index_sequences), maxnum):
        # Get a slice of elements for the current row
        row = index_sequences[i:i + maxnum]

        # If the row is shorter than maxnum, set values and mask for padding
        X[j, :len(row)] = row
        X[j, len(row):] = 1
        mask[j, :len(row)] = 1
        mask[j, len(row):] = 0

        j += 1

    return X, mask

def prepare_sentence_data(input):
    
    data = read_input_essay(input)

    X_data,mask = padding_sentence_sequences_input(data, max_num, post_padding=True)


    return X_data,mask

# from transformers import BertModel

file = 'bert-base-uncased'
bert_model = BertModel.from_pretrained(file)



class mlp(nn.Module):
    def __init__(self, in_f, out_f):
        super(mlp, self).__init__()
        self.layer1 = nn.Linear(in_f, 768)
        self.active = nn.Tanh()
        self.layer2 = nn.Linear(768, out_f)

    def forward(self, x):
        out = self.layer1(x)

        return out


class npcr_model(nn.Module):
    def __init__(self, maxSq=512):
        super(npcr_model, self).__init__()

        self.embedding = bert_model
        self.dropout = nn.Dropout(0.5)

        self.nn1 = nn.Linear(768, 768)
        self.output = nn.Linear(768, 1, bias=False)

        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias_ih' in name or 'bias_hh' in name)
        # nn.init.uniform(self.embed.weight.data, a=-0.5, b=0.5)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, x0, x1):
        x0_embed = self.embedding(x0)[1]
        x1_embed = self.embedding(x1)[1]

        # the linear layer nn1 can be replaced by MLP(the above or overwite by yourself)
        x0_nn1 = self.nn1(x0_embed)
        x1_nn1 = self.nn1(x1_embed)

        x0_nn1_d = self.dropout(x0_nn1)
        x1_nn1_d = self.dropout(x1_nn1)

        diff_x = (x0_nn1_d - x1_nn1_d)
        y = self.output(diff_x)

        y = torch.sigmoid(y)

        return y

model = npcr_model(512)

def replace_url(text):
    replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
    return replaced_text
max_num = 512
def is_number(token):
    return bool(num_regex.match(token))

model = torch.load('testrank_core_bert.prompt1.pt', map_location=torch.device('cpu'))
# model = torch.load('testrank_core_bert.prompt1.pt')
# model = model.to(device)

# Function to convert model prediction to a score
def convert_prediction_to_score(prediction):

    scaled_factor = scaled_value(0, 10)
    final_score = scaled_factor*prediction*2
    return final_score

def predict_score(essay_text, reference_text, model):
    # Prepare input data
    tensor_input,mask_input = prepare_sentence_data(essay_text)
    tensor_reference,mask_reference = prepare_sentence_data(reference_text)

    # Extract only the first sequence
    tensor_input = torch.tensor(tensor_input)
    mask_input = torch.tensor(mask_input)
    tensor_reference = torch.tensor(tensor_reference)
    mask_reference = torch.tensor(mask_reference)

    # Move tensors to the specified device
    tensor_input = tensor_input.to(device)
    mask_input = mask_input.to(device)
    tensor_reference = tensor_reference.to(device)
    mask_reference = mask_reference.to(device)

    size = tensor_input.size()

    # Use the loaded model to make predictions
    with torch.no_grad():
        prediction = model(tensor_input, tensor_reference)


    # Convert the model's prediction to a score
    predicted_score = convert_prediction_to_score(prediction[0].item())

    return predicted_score


# from Reference import list
lst =["the world. This is why chatting is the best setting. Computer helps the world in many ways. It helps kids if they need to research something. It helps the cops catch the bad people. The computer helps parents get from their children by vaction ads. This is has the computer helps the world. The computer does benifit our society. It has change the world. You can do many things on it. On the computer, you can chat with people from anywhere. The computer helps the world everyday. This is how the computer benifits our society.","Dear Newspaper, I think computers are great. There really helpful. They can teach us about things. Also they have fun games and websites which is always a plus. Computers are helpful in lots of things, like if you need to look something up for school you can just easily go on the computer and find it also some teachers have websites now where insted of bringing a big social studies book home, you can go on the internet and use the online book. Did you know that @PERCENT1 of @CAPS1 students rather use the internet book then taking the actual one home. Also if you forgot to write down your homework theres a website to lookup all the homework you have. Also computers can teach us things. There are websites that have math games and school related things. Also you can easilly go on google and look up things and they can give you thousands of information. Lastly the internet is a good source for fun they have millions of games on the internet and also fun websites like facebook. Did you know @PERCENT2 of @CAPS1 students use the computer for the games and online talking. This also gives kids lots of things to do on a rainy day. So weither your using the internet for it's helpfulness its resources, or just for the fun games and websites. Computers are really helpful.","Although some people believe that computers turn us children into zombies, I believe that they effect us in a posotive way. Computers can help us explore far away places that we @MONTH1 never go to. Maybe help us connect with an old friend or just a little help on homeworks. One reason why I think computers are only helping us is they help us explore othe places. I know for a fact that every kid dreams of a place that they would love to visit. But not at all of us can afford to travel far away. Most families however, do have computers. With these computers, kids with a parents permission, can use the internet to learn tons of cool facts about their place. ""I have always wanted to go to @LOCATION1 but my parents just don't have the proper incomg"" said @PERSON1 of @CAPS1-@CAPS2 @CAPS3 school @CAPS4 I used my @CAPS6 and the internet to discover. HUNDREDS of amazing facts on it."" @CAPS4 as you can see computers can help people realize their dreams. Another reason why I","said @PERSON1 of @CAPS1-@CAPS2 @CAPS3 school @CAPS4 I used my @CAPS6 and the internet to discover. HUNDREDS of amazing facts on it."" @CAPS4 as you can see computers can help people realize their dreams. Another reason why I think computers are good is they help you connect with friends. If you are like me, you like to talk to your friends. A @CAPS5! But sometimes they are too far away to call if you don't want to pay for it. A solution a new wonderful machine called. The @CAPS6! You can chat long distances without being for it. This is especially good for me because not too long ago, one of my best friends moved away to @LOCATION2. I thoughy I would never talk to him again. Then I learned about @CAPS7 @CAPS8. It allows you to talk whenever you want, whenever you want! Now, me and my friend can stay in touch and @CAPS4 can you! My finale reason as to why I think computers are good is they can help you on homework too. Even have a tricky math problem you couldn't figure out? How about the capital of a state that visit, slipped your mined? Well the @CAPS6 can solve both of those problems and more! @PERCENT1 of computers these days come with soft wear in them that comes with a calculator. For the other @PERCENT1 you can download and install softwear onto your @CAPS6 for a low price. And every @CAPS6 made in this world has internet capabilities. All's you need to do is it on and get ready for a ride! @CAPS4 don'y worry about too much @CAPS6 use, because computers can only help us not hurt us. @CAPS4 remember, computers help us live our dreams to the full extent. they also help us connect with no another, and help us excel in school. @CAPS4 get on, and get goin. Have you used your @CAPS6 today?","Dear local newspaper, @CAPS1 you know how long you go on the computer for? Well get up and go outside. Thats one of my reasons why I think people spend to much time on it my other reason is your eyesit can go bad. Its a nice sunny day dont wast your time on the computer! Go outside! All, kids @CAPS1 is sit on the computer all day not getting exercis. Did you know @NUM1 out of @NUM2 kids that sit on the computers all day gain at least @NUM3? @PERSON1 told me that. I mean I believe him because its true kids dont get up.There missing a nice day out would you wanna miss a nice sunny day? My finall reason is kids eyesight can go bad. I mean having them stare at a screen all day, thats bad. @PERSON1 told me if your on the computer for @NUM4 hours on the computer for @NUM4 hours straight your eyes can get worst. I remember a tiny when I was on the computer for @NUM4 1/@NUM4 hours my eyes were killing me everything I blink. I know the computer is fun playing games, talking to friends but you should take a break and @CAPS1 something ples. In conclusion just go outside and have fun! @CAPS1 you want to wast a nice sunny day by staying in doors hurting your eyes? I know I @CAPS1!"]

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loaded_model = model.to(device)


def vaibhav(input):
    score = []
    for i,reference_essay in enumerate(lst):
            predicted_score = predict_score(input, reference_essay, loaded_model)
            score.append(predicted_score)

    final =  ((sum(score)/len(score)))/10
    # print("Score of the essay is : ",final)
    return final


def get_final_score_vaibhav(essays):
    main_data = pd.DataFrame()
    essays = [essays]
    prompt = [prompt]


    main_data = pd.DataFrame()
    temp_df = pd.DataFrame({
        'score_vaibhav' : vaibhav(essays[0]),
        # 'normalized_score': normalized_score
      }, index=[0])

    main_data = pd.concat([main_data, temp_df], ignore_index=True)
    return main_data




import xgboost as xgb
model_path = 'xgboost_model.pth'
# bst.save_model(model_path)

# Load the model later
xgboost_model = xgb.Booster()
xgboost_model.load_model(model_path)


def get_final_score(essays, prompt):
    main_data = pd.DataFrame()
    essays = [essays]
    prompt = [prompt]

        # coherence_score_nli.append(coherence_score_snli(essays[index]))
    # print(type(essays[0]))
    means, variances, grammaticals, word_counts, char_counts, coherence_nli = get_mean_score(essays[0]), get_variance_score(essays[0]), get_grammatical_score(essays[0]),get_word_count(essays[0]), get_char_count(essays[0]), coherence_score_snli(essays[0])

    val = []
    for i in range(1):
        val.append(prompt[i]+"."+essays[i])


    combined_score = model_combined(essays,300).cpu().detach().numpy()


    val, lengths_batch = preprocess_essay(val)
    val = val.to(device)
    essays, lengths_batch = preprocess_essay(essays)
    essays = essays.to(device)
    out_semantic = model_semantic(essays, lengths_batch)
    out_coher = model_coher(essays, lengths_batch)

    out_prompt = model_prompt(val, lengths_batch)
    out_semantic = out_semantic.cpu().detach().numpy()
    out_coher = out_coher.cpu().detach().numpy()
    out_prompt = out_prompt.cpu().detach().numpy()

    # coherence_nli = coherence_score_nli(essay)
    # score_vaibhav = mittal_is_a_model(essay)

    main_data = pd.DataFrame()
    temp_df = pd.DataFrame({
        'means': means,
        'variances': variances,
        'grammaticals': grammaticals,
        'word_counts': word_counts,
        'char_counts': char_counts,
        'out_semantic': out_semantic[0],
        'out_coher': out_coher[0],
        'out_prompt': out_prompt[0],
        'coherence_nli' : coherence_nli,
        'combined_score' : combined_score[0][0],
        # 'score_vaibhav' : score_vaibhav,
        # 'normalized_score': normalized_score
      }, index=[0])

    main_data = pd.concat([main_data, temp_df], ignore_index=True)
    dpredict = xgb.DMatrix(main_data)
    predictions = xgboost_model.predict(dpredict)
    return predictions, main_data


string.punctuation
punct = string.punctuation
punct = punct.replace('.', '')
punct += '@'
'.' in punct

def get_lower(text):
    return text.lower()

def remove_punctuations(text):
    return ''.join([char for char in text if char not in punct])

def tokenize(text):
    # text = text.strip()
    return word_tokenize(text)

def remove_alpha_numeric(sentence):
    # return ' '.join(word for word in tokens if word.isalpha())
    words = sentence.split()
    alphabetic_words = [word for word in words if word.isalpha()]
    return ' '.join(alphabetic_words)

def remove_tags(text):
    return BeautifulSoup(text, 'html.parser').get_text()

def remove_extra_gaps(text):
    return ' '.join(text.split())

def pipeline(text):
    text = get_lower(text)
    text = remove_punctuations(text)
    # tokens = tokenize(text)
    # text = remove_alpha_numeric(text)
    # text = remove_tags(text)
    text = remove_extra_gaps(text)
    return text


import re

def is_human_generated(text):
    score=0
    words=text.split()

    # Feature : 1 sentence per paragraph
    if text.count('\n') >= 1:
        score=score+1
    
    # Feature : 1 word per paragraph
    if len(text.split()) <= 1:
        score=score+1
    
    # Feature : Specific characters present
    special_characters = [')', '-', ';', ':', '?']
    for word in words:
        if word in special_characters:
            score=score+1
    
    # Feature : Specific words present
    key_words = ['although', 'However', 'but', 'because', 'this', 'others', 'researchers', 'et']
    for word in words:
        if word in key_words:
            score=score+1  # Human-generated
    
    # Feature : Contains ‘‘’‘‘ (single quotes)
    if "’‘’‘" in text:
        score += 1 
    
    # Feature : Standard deviation in sentence length
    sentence_lengths = [len(sentence.split()) for sentence in re.split(r'[.!?]', text) if sentence.strip()]
    if len(sentence_lengths) >= 2 and (max(sentence_lengths) - min(sentence_lengths)) > 1:
        score=score+1
    
    # Feature : Contains numbers
    if any(char.isdigit() for char in text):
        score=score+1
    
    # Feature : Contains 2 times more capitals than ‘‘.’’
    if text.count('.') > 0 and text.count('.') * 2 < sum(1 for char in text if char.isupper()):
        score=score+1  # Human-generated
    
    threshold=3
    # If none of the individual features are present, consider it AI-generated
    return score<=threshold

# Example usage:
# text_to_check = "This is a sample text written by a human."
# if is_human_generated(text_to_check):
#     print("Human-generated")
# else:
#     print("AI-generated")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        essay_text = request.form.get('essay_text', '')
        prompt = request.form.get('prompt', '')
        essay_text = pipeline(essay_text)
        prompt = pipeline(prompt)
        # print("="*80)
        # print(essay_text)
        # print("="*80)
        # print(prompt)
        # print("="*80)
        # Assuming `get_final_score` returns a NumPy array
        output, _ = get_final_score(essay_text, prompt)
        print(output)
        if(is_human_generated(essay_text)): 
            ans = "HUMAN"
        else:
            ans = "AI"
            
        temp = "Essays text is " + ans + " generated"
        output = str(np.round(output*10)) + "\n" + temp 
        # Convert NumPy array to Python list
        output_list = output
        return jsonify({'prediction': output_list})


if __name__ == '__main__':
    app.run(debug=True)
