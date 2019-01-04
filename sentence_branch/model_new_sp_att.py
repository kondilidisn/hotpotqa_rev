# -*- coding: utf-8 -*-


import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn
import os

class SPModel(nn.Module):
    def __init__(self, config, word_mat, char_mat):
        super().__init__()
        self.config = config
        self.word_dim = config.glove_dim
        self.word_emb = nn.Embedding(len(word_mat), len(word_mat[0]), padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(word_mat))
        self.word_emb.weight.requires_grad = False
        self.char_emb = nn.Embedding(len(char_mat), len(char_mat[0]), padding_idx=0)
        self.char_emb.weight.data.copy_(torch.from_numpy(char_mat))

        self.char_cnn = nn.Conv1d(config.char_dim, config.char_hidden, 5)
        self.char_hidden = config.char_hidden
        self.hidden = config.hidden

        self.rnn = EncoderRNN(config.char_hidden+self.word_dim, config.hidden, 1, True, True, 1-config.keep_prob, False,config.save)

        self.qc_att = BiAttention(config.hidden*2, 1-config.keep_prob)
        self.linear_1 = nn.Sequential(
                nn.Linear(config.hidden*8, config.hidden),
                nn.ReLU()
            )

        self.rnn_2 = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1-config.keep_prob, False,config.save)
        
        self.rnn_sentence = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1-config.keep_prob, False,config.save)
        self.self_att_sentences = BiAttention(config.hidden, 1-config.keep_prob)
        self.linear_sent_att = nn.Sequential(
                nn.Linear(config.hidden*8, config.hidden),
                nn.ReLU()
            )
        
        self.self_att = BiAttention(config.hidden*2, 1-config.keep_prob)
        self.linear_2 = nn.Sequential(
                nn.Linear(config.hidden*8, config.hidden),
                nn.ReLU()
            )

        self.rnn_sp = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1-config.keep_prob, False,config.save)
        self.linear_sp = nn.Linear(config.hidden*4, 1)

#        self.rnn_start = EncoderRNN(config.hidden*3+80, config.hidden, 1, False, True, 1-config.keep_prob, False,config.save)
#        self.linear_start = nn.Linear(config.hidden*2, 1)
#
#        self.rnn_end = EncoderRNN(config.hidden*3, config.hidden, 1, False, True, 1-config.keep_prob, False,config.save)
#        self.linear_end = nn.Linear(config.hidden*2, 1)
#
#        self.rnn_type = EncoderRNN(config.hidden*3, config.hidden, 1, False, True, 1-config.keep_prob, False,config.save)
#        self.linear_type = nn.Linear(config.hidden*2, 3)

        self.cache_S = 0

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def mean_pooling_module(self,all_mapping,word_embeting_array,save):
    
        sentence_embeddings = torch.matmul(all_mapping.permute(0, 2, 1).contiguous(), word_embeting_array)

        sentence_length = torch.sum(all_mapping.permute(0, 2, 1),dim=2)

        sentence_mask = (sentence_length > 0).float()

        sentence_length = sentence_length.unsqueeze(2)
#        logging("Sentence length unsqueeze shape : "+str(sentence_length.shape),self.config.save)
#        logging("Sentence length unsqueeze : "+str(sentence_length),self.config.save)
        sentence_length = sentence_length.repeat(1,1,self.hidden)
#        logging("Sentence length repeat shape : "+str(sentence_length.shape),self.config.save)
#        logging("Sentence length repeat : "+str(sentence_length),self.config.save)
        
        sentence_length[sentence_length==0] = 1
        
        sentence_results = torch.div(sentence_embeddings,sentence_length)
#        logging("Result embeddings : "+str(sentence_results.shape),self.config.save)
                
        return sentence_results,sentence_mask
        
#        sentences = torch.zeros(start_mapping.size(0),start_mapping.size(2),word_embeting_array.size(2))
#        sentences = sentences.cuda()
#        for b in range(start_mapping.size(0)):
#            # for every sentence
#            for i in range(start_mapping.size(2)):
##                logging("start of the sentence "+str(torch.argmax(start_mapping[b,:,i])),save)
##                logging("end of the sentence "+str(torch.argmax(end_mapping[b,:,i])),save)
#                start_index = torch.argmax(start_mapping[b,:,i])
#                end_index = torch.argmax(end_mapping[b,:,i])
#                sentence_embeting = word_embeting_array[b,start_index,:]
#                
#                for index in range(start_index+1,end_index+1):
#                    sentence_embeting+=word_embeting_array[b,index,:]
#                num = end_index-start_index+1
#                
#                if int(num.cpu().data.numpy())==1:
#                    break
#                torch.div(sentence_embeting,int(num.cpu().data.numpy()))
#                sentences[b,i,:] = sentence_embeting
#        
        

    def forward(self, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=False):
        para_size, ques_size, char_size, bsz = context_idxs.size(1), ques_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)

        context_mask = (context_idxs > 0).float()
        ques_mask = (ques_idxs > 0).float()

        context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
        ques_ch = self.char_emb(ques_char_idxs.contiguous().view(-1, char_size)).view(bsz * ques_size, char_size, -1)

        context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)
        ques_ch = self.char_cnn(ques_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, ques_size, -1)

        context_word = self.word_emb(context_idxs)
        ques_word = self.word_emb(ques_idxs)
        
        
        
        
#        logging("Start",self.config.save)
#        logging("context word "+str(context_word.shape),self.config.save)
#        logging("context ch"+str(context_ch.shape),self.config.save)
#        logging("question word"+str(ques_word.shape),self.config.save)
#        logging("question ch"+str(context_word.shape),self.config.save)
        
        context_output = torch.cat([context_word, context_ch], dim=2)
        ques_output = torch.cat([ques_word, ques_ch], dim=2)

#        logging("context output"+str(context_output.shape),self.config.save)
#        logging("ques output"+str(ques_output.shape),self.config.save)
#        
        context_output = self.rnn(context_output, context_lens)
        ques_output = self.rnn(ques_output)

#        logging("context output after RNN"+str(context_output.shape),self.config.save)
#        logging("ques output after RNN"+str(ques_output.shape),self.config.save)
#        
        output = self.qc_att(context_output, ques_output, ques_mask)
#        logging("attension output"+str(output.shape),self.config.save)
        output = self.linear_1(output)
#        logging("attension linear output"+str(output.shape),self.config.save)
        
        # Sentence branch
        sentence_embeddings , sentence_mask = self.mean_pooling_module(all_mapping ,output ,self.config.save)
#        logging("sentence embeddings "+str(sentence_embeddings.shape),self.config.save)
        
#        sentence_rnn_out = self.rnn_sentence(sentence_embeddings)
#        logging("sentence rnn output "+str(sentence_rnn_out.shape),self.config.save)
#        logging("sentece mask "+str(sentence_mask.shape),self.config.save)
        
        sentence_self_attension_out = self.self_att_sentences(sentence_embeddings,sentence_embeddings,sentence_mask)
#        logging("sentence self output "+str(sentence_self_attension_out.shape),self.config.save)
#        self.self_att_sentences
        
        sentence_embeddings = self.linear_sent_att(sentence_self_attension_out)
        
        sentence_embeddings = self.rnn_sentence(sentence_embeddings)
        
        # end
        output_t = self.rnn_2(output, context_lens)
#        logging("context lens "+str(context_lens.shape),self.config.save)
#        logging("output size " + str(output.shape),self.config.save)
#        logging("output 2nd RNN"+str(output_t.shape),self.config.save)
        output_t = self.self_att(output_t, output_t, context_mask)
#        logging("context_mask "+str(context_mask.shape),self.config.save)
#        logging("mask example 1 "+str(torch.sum(context_mask[0,:])),self.config.save)
#        logging("mask example 2 "+str(torch.sum(context_mask[1,:])),self.config.save)
#        logging("mask example 3 "+str(torch.sum(context_mask[2,:])),self.config.save)
#        logging("mask example 4 "+str(torch.sum(context_mask[3,:])),self.config.save)

#        logging("self attension output"+str(output_t.shape),self.config.save)
        output_t = self.linear_2(output_t)
#        logging("linear after self attension"+str(output_t.shape),self.config.save)
        
        output = output + output_t
#        logging("sum bi-attension and self-attension"+str(output.shape),self.config.save)
        sp_output = self.rnn_sp(output, context_lens)
#        logging("supporting fact RNN"+str(output.shape),self.config.save)
        
        start_output = torch.matmul(start_mapping.permute(0, 2, 1).contiguous(), sp_output[:,:,self.hidden:])
#        logging("start output sp"+str(start_output.shape),self.config.save)
        end_output = torch.matmul(end_mapping.permute(0, 2, 1).contiguous(), sp_output[:,:,:self.hidden])
#        logging("end output sp"+str(end_output.shape),self.config.save)
        sp_output = torch.cat([start_output, end_output], dim=-1)
#        logging("sp output "+str(sp_output.shape),self.config.save)
        
        sp_output = torch.cat([sp_output, sentence_embeddings], dim=-1)
#        logging("sp output sentence "+str(sp_output_sentence.shape),self.config.save)
        
        sp_output_t = self.linear_sp(sp_output)
#        logging("sp output after linear"+str(sp_output.shape),self.config.save)
        sp_output_aux = Variable(sp_output_t.data.new(sp_output_t.size(0), sp_output_t.size(1), 1).zero_())
#        logging("sp output aux"+str(sp_output_aux.shape),self.config.save)
       
        predict_support = torch.cat([sp_output_aux, sp_output_t], dim=-1).contiguous()
#        logging("predict_support"+str(predict_support.shape),self.config.save)
        
        return predict_support

def logging(s,save, print_=True, log_=True):
    with open(os.path.join(save, 'log.txt'), 'a+') as f_log:
        f_log.write(s + '\n')
                
class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x

class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last, save):
        super().__init__()
        self.save=save
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()


    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        counter=1
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
#            logging("print output before dropout "+str(output.shape),self.save)
            output = self.dropout(output)
#            logging("print output after dropout "+str(output.shape),self.save)
            if input_lengths is not None:
#                logging("lens shape"+str(lens.shape)+" in iteration "+str(counter),self.save)
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)
#                logging("output with lens : "+str(len(output))+","+str(len(output[0])),self.save)
#                logging("output 0 : "+str(output[0].shape),self.save)
#                logging("output 1 : "+str(output[1].shape),self.save)
#                logging("output : "+str(output),self.save)
            output, hidden = self.rnns[i](output, hidden)
#            logging("RNN output "+str(len(output)),self.save)
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
#                logging("final output "+str(output.shape)+" in iteration "+str(counter),self.save)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
            counter+=1
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]

class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)

class GateLayer(nn.Module):
    def __init__(self, d_input, d_output):
        super(GateLayer, self).__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.gate = nn.Linear(d_input, d_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.linear(input) * self.sigmoid(self.gate(input))

