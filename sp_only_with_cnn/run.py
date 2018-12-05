




import ujson as json
import numpy as np
from tqdm import tqdm
import os
from torch import optim, nn
from model import Model #, NoCharModel, NoSelfModel
from sp_model import SPModel
# from normal_model import NormalModel, NoSelfModel, NoCharModel, NoSentModel
# from oracle_model import OracleModel, OracleModelV2
# from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset
from util import convert_tokens, evaluate
from util import get_buckets, DataIterator, IGNORE_INDEX
import time
import shutil
import random
import torch
from torch.autograd import Variable
import sys
from torch.nn import functional as F

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

nll_sum = nn.CrossEntropyLoss(reduction='sum', ignore_index=IGNORE_INDEX)
w=torch.from_numpy(np.array([ 0.04, 1.], dtype=float)).float().cuda()
nll_average = nn.CrossEntropyLoss(weight=w, reduction='elementwise_mean', ignore_index=IGNORE_INDEX)
nll_all = nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)

def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=['run.py', 'model.py', 'util.py', 'sp_model.py'])
    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    logging('Config')
    for k, v in config.__dict__.items():
        logging('    - {} : {}'.format(k, v))

    logging("Building model...")
    train_buckets = get_buckets(config.train_record_file)
    dev_buckets = get_buckets(config.dev_record_file)

    def build_train_iterator():
        return DataIterator(train_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, True, config.sent_limit)

    def build_dev_iterator():
        return DataIterator(dev_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, False, config.sent_limit)

    if config.sp_lambda > 0:
        model = SPModel(config, word_mat, char_mat)
    else:
        model = Model(config, word_mat, char_mat)

    logging('nparams {}'.format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))
    #ori_model = model.cuda()
    #ori_model = model
    #model = nn.DataParallel(ori_model)
    model = model.cuda()

    lr = config.init_lr
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)
    cur_patience = 0
    total_loss = 0
    global_step = 0
    best_dev_F1 = None
    stop_train = False
    start_time = time.time()
    eval_start_time = time.time()
    model.train()

    train_metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}

    best_dev_sp_f1 = 0

    # total_support_facts = 0
    # total_contexes = 0
    total_support_facts =0
    total_contexes = 0

    for epoch in range(5):
        for data in build_train_iterator():
            context_idxs = Variable(data['context_idxs'])
            ques_idxs = Variable(data['ques_idxs'])
            context_char_idxs = Variable(data['context_char_idxs'])
            ques_char_idxs = Variable(data['ques_char_idxs'])
            context_lens = Variable(data['context_lens'])

            y1 = Variable(data['y1'])
            y2 = Variable(data['y2'])
            q_type = Variable(data['q_type'])
            
            is_support = Variable(data['is_support'])


            start_mapping = Variable(data['start_mapping'])
            end_mapping = Variable(data['end_mapping'])
            all_mapping = Variable(data['all_mapping'])

            # print(all_mapping.size())

            # total_support_facts += torch.sum(torch.sum(is_support))
            # total_contexes += is_support.size(0) * is_support.size(1)

            # context_idxs : torch.Size([1, 1767])
            # ques_idxs : torch.Size([1, 17])
            # context_char_idxs : torch.Size([1, 1767, 16])
            # ques_char_idxs : torch.Size([1, 17, 16])
            # context_lens : tensor([1767])

            # start_mapping : torch.Size([1, 1767, 65])
            # end_mapping : torch.Size([1, 1767, 65])
            # all_mapping : torch.Size([1, 1767, 65])


            # continue

            # change the input format into Sentences (input) -> Is support fact (target)


            # get total number of sentences 
            number_of_sentences = int(torch.sum(torch.sum(start_mapping, dim = 1)).item())
            #print('number_of_sentences=', number_of_sentences)
            # get sentence limit
            sentence_limit = config.sent_limit
            # get question limit
            ques_limit = config.ques_limit
            # get character limit
            char_limit = config.char_limit
            sent_limit = 600


            # allocate space
            sentence_idxs = torch.zeros(number_of_sentences, sent_limit, dtype = torch.long).cuda()
            sentence_char_idxs = torch.zeros(number_of_sentences, sent_limit, char_limit, dtype = torch.long).cuda()
            sentence_lengths = torch.zeros(number_of_sentences, dtype = torch.long).cuda()
            is_support_fact = torch.zeros(number_of_sentences, dtype = torch.long).cuda()


            ques_idxs_sen_impl = torch.zeros(number_of_sentences, ques_limit, dtype = torch.long).cuda()
            ques_char_idxs_sen_impl = torch.zeros(number_of_sentences, ques_limit, char_limit, dtype = torch.long).cuda()


            # sentence_idxs = []
            # sentence_char_idxs = []
            # sentence_lengths = []
            # is_support_fact = []

            # ques_idxs_sen_impl = []
            # ques_char_idxs_sen_impl = []

            index = 0

            # for every batch
            for b in range(all_mapping.size(0)):
                # for every sentence
                for i in range(all_mapping.size(2)):
                    # get sentence map
                    sentence_map = all_mapping[b,:, i]
                    s = torch.sum(sentence_map)
                    # if there are no more sentences on this batch (but only zero-pads) then continue to next batch
                    if s == 0:
                        continue

                    # get sentence

                    # get starting index
                    starting_index = torch.argmax(start_mapping[b,:,i])
                    # get ending index
                    ending_index = torch.argmax(end_mapping[b,:,i]) + 1
                    # get sentence length
                    sentence_length = int(torch.sum(all_mapping[b,:,i]))
                    sentence = context_idxs[b, starting_index: ending_index]
                    sentence_chars = context_char_idxs[b, starting_index: ending_index, :]
                    #print('sentence=', sentence)

                    #if sentence_length>100: 
                    #    print('Sentence starts :', starting_index, ', & end :', ending_index, ', Total tokens sentence :', torch.sum(all_mapping[b,:,i]), 'sentence_length=', sentence_length, 'start mapping=',start_mapping[b,:,i], 'end mapping=', end_mapping[b,:,i])
                    #    os.system("pause")
					
                    sentence_idxs[index, :sentence_length] = sentence
                    sentence_char_idxs[index, :sentence_length, :] = sentence_chars
                    sentence_lengths[index] = sentence_length
                    is_support_fact[index] = is_support[b,i]
                    # repeat for the question

                    ques_idxs_sen_impl[index, : ques_idxs[b,:].size(0) ] = ques_idxs[b,:]
                    ques_char_idxs_sen_impl[index, : ques_idxs[b,:].size(0), : ] = ques_char_idxs[b,:,:]
                    # append to lists
                    # sentence_idxs.append(sentence)
                    # sentence_char_idxs.append(sentence_chars)
                    # sentence_lengths.append(sentence_length)
                    # is_support_fact.append(is_support[b,i])

                    # repeat for the question
                    # ques_idxs_sen_impl.append(ques_idxs[b,:])
                    # ques_char_idxs_sen_impl.append(ques_char_idxs[b,:,:])

                    index += 1


            # zero padd
            sentence_length = torch.max(sentence_lengths)

            # torch.Tensor()

            # for i in range(len(sentence_idxs)):


            
            # sentence_idxs = torch.stack(sentence_idxs)
            # sentence_char_idxs = torch.stack(sentence_char_idxs)
            # sentence_lengths = torch.stack(sentence_lengths)
            # is_support_fact = torch.stack(is_support_fact)

            # ques_idxs_sen_impl = torch.stack(ques_idxs_sen_impl)
            # ques_char_idxs_sen_impl = torch.stack(ques_char_idxs_sen_impl)






            # predict_support = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=False)

            predict_support = model(sentence_idxs, ques_idxs_sen_impl, sentence_char_idxs, ques_char_idxs_sen_impl, sentence_length, start_mapping, end_mapping)

            # logit1, logit2, predict_type, predict_support = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=False)
            # loss_1 = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0)
            #loss_2 = nll_average(predict_support.view(-1, 2), is_support.view(-1))
            #print('predict_support sz=',predict_support.size(), 'is_support_fact sz=', is_support_fact.size(), 'is_support_fact.unsqueeze=', is_support_fact.unsqueeze(1).size())
            loss_2 = nll_average(predict_support.contiguous(), is_support_fact.contiguous())
            # loss = loss_1 + config.sp_lambda * loss_2
            loss = loss_2

            # update train metrics
#            train_metrics = update_sp(train_metrics, predict_support.view(-1, 2), is_support.view(-1))
            train_metrics = update_sp(train_metrics, predict_support, is_support_fact)

            #exit()

            # ps = predict_support.view(-1, 2)
            # iss = is_support.view(-1)

            # print('Predicted SP output  and ground truth:')
            # length = predict_support.view(-1, 2).shape[0]
            # for jj in range(length):
            #     print(ps[jj,1] , '   :   ', iss[jj])

            # temp = torch.cat([ predict_support.view(-1, 2).float(), is_support.view(-1)], dim=-1).contiguous()
            # print(temp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            total_loss += loss.data.item()
            global_step += 1

            if global_step % config.period == 0:

                # avegage metrics
                for key in train_metrics:
                    train_metrics[key] /= float(config.period)

                # cur_loss = total_loss / config.period
                cur_loss = total_loss

                elapsed = time.time() - start_time
                logging('| epoch {:3d} | step {:6d} | lr {:05.5f} | ms/batch {:5.2f} | train loss {:8.3f} | SP EM {:8.3f} | SP f1 {:8.3f} | SP Prec {:8.3f} | SP Recall {:8.3f}'.format(epoch, global_step, lr, elapsed*1000/config.period, cur_loss, train_metrics['sp_em'], train_metrics['sp_f1'], train_metrics['sp_prec'], train_metrics['sp_recall']))

                total_loss = 0
                train_metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}
                start_time = time.time()

            if global_step % config.checkpoint == 0:
                model.eval()

                # metrics = evaluate_batch(build_dev_iterator(), model, 5, dev_eval_file, config)
                eval_metrics = evaluate_batch(build_dev_iterator(), model, 500, dev_eval_file, config)
                model.train()

                logging('-' * 89)
                # logging('| eval {:6d} in epoch {:3d} | time: {:5.2f}s | dev loss {:8.3f} | EM {:.4f} | F1 {:.4f}'.format(global_step//config.checkpoint,
                #     epoch, time.time()-eval_start_time, metrics['loss'], metrics['exact_match'], metrics['f1']))
                logging('| eval {:6d} in epoch {:3d} | time: {:5.2f}s | dev loss {:8.3f}| SP EM {:8.3f} | SP f1 {:8.3f} | SP Prec {:8.3f} | SP Recall {:8.3f}'.format(global_step//config.checkpoint,
                    epoch, time.time()-eval_start_time, eval_metrics['loss'], eval_metrics['sp_em'], eval_metrics['sp_f1'], eval_metrics['sp_prec'], eval_metrics['sp_recall']))
                logging('-' * 89)

                if eval_metrics['sp_f1'] > best_dev_sp_f1:
                    best_dev_sp_f1 = eval_metrics['sp_f1']
                    torch.save(model.state_dict(), os.path.join(config.save, 'model.pt'))
                    cur_patience = 0
                else:
                    cur_patience += 1
                    if cur_patience >= config.patience:
                        lr *= 0.75
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        if lr < config.init_lr * 1e-2:
                            stop_train = True
                            break
                        cur_patience = 0

                eval_start_time = time.time()


        total_support_facts += torch.sum(torch.sum(is_support))
        total_contexes += is_support.size(0) * is_support.size(0)

        # print('total_support_facts :', total_support_facts)
        # print('total_contexes :', total_contexes)
        # exit()



        if stop_train: break
    logging('best_dev_F1 {}'.format(best_dev_sp_f1))

def update_sp(metrics, predict_support, is_support):
    tp, fp, fn = 0, 0, 0
    total = is_support.shape[0]

    predict_support = torch.nn.functional.softmax(predict_support, dim=1)
    #print('total=', total)
    for i in range (total-1):
        #print('i=', i)
        # print('Prediction :', predict_support[i,1].item(), ', Ground truth :',  is_support[i].item())
        predicted = predict_support[i,1] >= predict_support[i,0]
        ground_truth = (is_support[i] == 1)
        #print('predicted=',predicted, 'ground_truth=',ground_truth)
        
        if predicted:
            # true positive
            if ground_truth:
                tp += 1
            # false positive
            else:
                fp += 1
        else:
            # false negative
            if ground_truth:
                fn +=1
        # print(tp , fp, fn)

    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return metrics

def get_support_fact_accuracy(predict_support, is_support):
    correct_counter = 0
    total = is_support.shape[0]

    # print(predict_support)
    # print(is_support)

    for i in range (total):
        if (predict_support[i,1] > 0 and is_support[i] == 1 ) or (predict_support[i,1] < 0 and is_support[i] == 0 ):
            correct_counter += 1

    return correct_counter / float(total)



def evaluate_batch(data_source, model, max_batches, eval_file, config):
    answer_dict = {}
    sp_dict = {}
    total_loss, step_cnt = 0, 0
    iter = data_source
    eval_metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}

    for step, data in enumerate(iter):
        if step >= max_batches and max_batches > 0: break

        context_idxs = Variable(data['context_idxs'], volatile=True)
        ques_idxs = Variable(data['ques_idxs'], volatile=True)
        context_char_idxs = Variable(data['context_char_idxs'], volatile=True)
        ques_char_idxs = Variable(data['ques_char_idxs'], volatile=True)
        context_lens = Variable(data['context_lens'], volatile=True)
        y1 = Variable(data['y1'], volatile=True)
        y2 = Variable(data['y2'], volatile=True)
        q_type = Variable(data['q_type'], volatile=True)
        is_support = Variable(data['is_support'], volatile=True)
        start_mapping = Variable(data['start_mapping'], volatile=True)
        end_mapping = Variable(data['end_mapping'], volatile=True)
        all_mapping = Variable(data['all_mapping'], volatile=True)
        number_of_sentences = int(torch.sum(torch.sum(start_mapping, dim = 1)).item())
        #print('number_of_sentences=', number_of_sentences)
        # get sentence limit
        sentence_limit = config.sent_limit
        # get question limit
        ques_limit = config.ques_limit
        # get character limit
        char_limit = config.char_limit
        sent_limit=600


        # allocate space
        sentence_idxs = torch.zeros(number_of_sentences, sent_limit, dtype = torch.long).cuda()
        sentence_char_idxs = torch.zeros(number_of_sentences, sent_limit, char_limit, dtype = torch.long).cuda()
        sentence_lengths = torch.zeros(number_of_sentences, dtype = torch.long).cuda()
        is_support_fact = torch.zeros(number_of_sentences, dtype = torch.long).cuda()


        ques_idxs_sen_impl = torch.zeros(number_of_sentences, ques_limit, dtype = torch.long).cuda()
        ques_char_idxs_sen_impl = torch.zeros(number_of_sentences, ques_limit, char_limit, dtype = torch.long).cuda()


        # sentence_idxs = []
        # sentence_char_idxs = []
        # sentence_lengths = []
        # is_support_fact = []

        # ques_idxs_sen_impl = []
        # ques_char_idxs_sen_impl = []

        index = 0

        # for every batch
        for b in range(all_mapping.size(0)):
            # for every sentence
            for i in range(all_mapping.size(2)):
                # get sentence map
                sentence_map = all_mapping[b,:, i]
                s = torch.sum(sentence_map)
                # if there are no more sentences on this batch (but only zero-pads) then continue to next batch
                if s == 0:
                    continue

                # get sentence

                # get starting index
                starting_index = torch.argmax(start_mapping[b,:,i])
                # get ending index
                ending_index = torch.argmax(end_mapping[b,:,i]) + 1
                # get sentence length
                sentence_length = int(torch.sum(all_mapping[b,:,i]))
                #print('Sentence starts :', starting_index, ', & end :', ending_index, ', Total chars :', torch.sum(all_mapping[b,:,i]))
                sentence = context_idxs[b, starting_index: ending_index]
                sentence_chars = context_char_idxs[b, starting_index: ending_index, :]

                #print('index=', index, 'sentence_length=', sentence_length, 'sentence length=', sentence.size())

                sentence_idxs[index, :sentence_length] = sentence
                sentence_char_idxs[index, :sentence_length, :] = sentence_chars
                sentence_lengths[index] = sentence_length
                is_support_fact[index] = is_support[b,i]
                # repeat for the question

                ques_idxs_sen_impl[index, : ques_idxs[b,:].size(0) ] = ques_idxs[b,:]
                ques_char_idxs_sen_impl[index, : ques_idxs[b,:].size(0), : ] = ques_char_idxs[b,:,:]
                # append to lists
                # sentence_idxs.append(sentence)
                # sentence_char_idxs.append(sentence_chars)
                # sentence_lengths.append(sentence_length)
                # is_support_fact.append(is_support[b,i])

                # repeat for the question
                # ques_idxs_sen_impl.append(ques_idxs[b,:])
                # ques_char_idxs_sen_impl.append(ques_char_idxs[b,:,:])

                index += 1


        # zero padd
        sentence_length = torch.max(sentence_lengths)

        predict_support= model(sentence_idxs, ques_idxs_sen_impl, sentence_char_idxs, ques_char_idxs_sen_impl, sentence_length, start_mapping, end_mapping)
        
        # update eval metrics
        eval_metrics = update_sp(eval_metrics, predict_support, is_support_fact)

        # logit1, logit2, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
        # loss = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0) + config.sp_lambda * nll_average(predict_support.view(-1, 2), is_support.view(-1))
        
        loss =  nll_average(predict_support, is_support_fact)
        
        # answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        # answer_dict.update(answer_dict_)

        total_loss += loss.data.item()
        step_cnt += 1


    # avegage metrics
    for key in eval_metrics:
        eval_metrics[key] /= float(step_cnt)

    eval_metrics['loss'] = total_loss / step_cnt

    # return metrics
    return eval_metrics

def predict(data_source, model, eval_file, config, prediction_file):
    answer_dict = {}
    sp_dict = {}
    sp_th = config.sp_threshold


    eval_metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}

    for step, data in enumerate(tqdm(data_source)):
        context_idxs = Variable(data['context_idxs'], volatile=True)
        ques_idxs = Variable(data['ques_idxs'], volatile=True)
        context_char_idxs = Variable(data['context_char_idxs'], volatile=True)
        ques_char_idxs = Variable(data['ques_char_idxs'], volatile=True)
        context_lens = Variable(data['context_lens'], volatile=True)
        start_mapping = Variable(data['start_mapping'], volatile=True)
        end_mapping = Variable(data['end_mapping'], volatile=True)
        all_mapping = Variable(data['all_mapping'], volatile=True)

        # logit1, logit2, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
        # answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        # answer_dict.update(answer_dict_)

        predict_support = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)

        eval_metrics = update_sp(eval_metrics, predict_support.view(-1, 2), is_support.view(-1))


        predict_support_np = torch.sigmoid(predict_support[:, :, 1]).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = data['ids'][i]
            for j in range(predict_support_np.shape[1]):
                if j >= len(eval_file[cur_id]['sent2title_ids']): break
                if predict_support_np[i, j] > sp_th:
                    cur_sp_pred.append(eval_file[cur_id]['sent2title_ids'][j])
            sp_dict.update({cur_id: cur_sp_pred})

    # prediction = {'answer': answer_dict, 'sp': sp_dict}
    prediction = {'sp': sp_dict}
    with open(prediction_file, 'w') as f:
        json.dump(prediction, f)

def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    if config.data_split == 'dev':
        with open(config.dev_eval_file, "r") as fh:
            dev_eval_file = json.load(fh)
    else:
        with open(config.test_eval_file, 'r') as fh:
            dev_eval_file = json.load(fh)
    with open(config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    if config.data_split == 'dev':
        dev_buckets = get_buckets(config.dev_record_file)
        para_limit = config.para_limit
        ques_limit = config.ques_limit
    elif config.data_split == 'test':
        para_limit = None
        ques_limit = None
        dev_buckets = get_buckets(config.test_record_file)

    def build_dev_iterator():
        return DataIterator(dev_buckets, config.batch_size, para_limit,
            ques_limit, config.char_limit, False, config.sent_limit)

    if config.sp_lambda > 0:
        model = SPModel(config, word_mat, char_mat)
    else:
        model = Model(config, word_mat, char_mat)
    # ori_model = model.cuda()
    ori_model = model
    ori_model.load_state_dict(torch.load(os.path.join(config.save, 'model.pt')))
    model = ori_model
    # model = nn.DataParallel(ori_model)

    model.eval()
    predict(build_dev_iterator(), model, dev_eval_file, config, config.prediction_file)

