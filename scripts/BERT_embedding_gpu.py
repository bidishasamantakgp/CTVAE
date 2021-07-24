import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import sys
import spacy
from tqdm import tqdm
from collections import defaultdict
import re
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
#import logging
#YELP
#MAXLEN = 22
#AMAZON
#MAXLEN = 47
#IMDB
MAXLEN = 37 
def get_bert_embedding(text, nlp, model, tokenizer):
    sentence_embedding = []
    marked_text = "[CLS] " + text.strip() + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    position_ids = range(len(tokenized_text))

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).cuda()
    segments_tensors = torch.tensor([segments_ids]).cuda()
    position_tensors = torch.tensor([position_ids]).cuda()

        
    with torch.no_grad():
            model.cuda()
            encoded_layers, _ = model(tokens_tensor, segments_tensors)

    token_embeddings = torch.stack(encoded_layers, dim=0).cuda()
    token_embeddings = torch.squeeze(token_embeddings, dim=1).cuda()
    token_embeddings = token_embeddings.permute(1,0,2).cuda()
    token_vecs_sum = []
    
    token_vecs = encoded_layers[11][0]
    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0).cuda()
      

    return sentence_embedding


if __name__=="__main__":
    a = open(sys.argv[1])
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    nlp = spacy.load("en_core_web_sm")
    s_e_list = list()
    #w_e_list = list()
    count = 0
    for line in tqdm(a):
        count += 1

        s_e = get_bert_embedding(line, nlp, model, tokenizer)
        if s_e is not None:
         s_e_list.append(s_e)
        #if count == 128:
        #    break
    print(len(s_e_list))
    torch.save(torch.stack(s_e_list), sys.argv[2])
    #torch.save(s_e, 'sentence-10.pt')
    #torch.save(w_e, 'word-10.pt')

    #torch.save(s_e, 'sentence-all.pt')
    #torch.save(w_e, 'word-all.pt')
