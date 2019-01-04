




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
from pathlib import Path



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
nll_average = nn.CrossEntropyLoss(reduction='elementwise_mean', ignore_index=IGNORE_INDEX)
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
    # train_buckets = dev_buckets

    def build_train_iterator():
        return DataIterator(train_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, True, config.sent_limit)

    def build_dev_iterator():
        return DataIterator(dev_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, False, config.sent_limit)

    if config.sp_lambda > 0:
        model = SPModel(config, word_mat, char_mat)
    else:
        model = Model(config, word_mat, char_mat)

    logging('nparams {}'.format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))
    ori_model = model.cuda()
    # ori_model = model

    # flag (checking if the learning will be loaded from the file or not)
    # lr = 0



    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)

    # if the training was interrupted, then we load last trained state of the model
    logging('Checking if previous training was interrupted...')


    my_file = Path('temp_model')
    if my_file.exists():
        logging('Previous training was interupted, loading model and optimizer state')
        ori_model.load_state_dict(torch.load('temp_model'))
        optimizer.load_state_dict(torch.load('temp_optimizer'))

        # training_state = training_state.load_whole_class('last_training_state.pickle')
        # training_state = training_state.load_whole_class('last_training_state.pickle')

    # # if os.path.exists('last_model.pt'):
    # for dp, dn, filenames in os.walk('.'):
    #     model_filenames = []
    #     corresponding_learning_rates = []
    #     for ff in filenames:
    #         if ff.endswith("last_model.pt"):
    #             # putting all found models on list
    #             lr = float(ff.split('_')[0])
    #             model_filenames.append(ff)
    #             corresponding_learning_rates.append(lr)

    #     if len( model_filenames) > 0:
    #         # selecting the model with the smallest learning rate to be loaded
    #         loading_model_index = np.argmin(corresponding_learning_rates)
    #         # continuing with the previous learning rate
    #         lr = corresponding_learning_rates[loading_model_index]

    #         logging('Previous training was interrupted so loading last saved state model and continuing.')
    #         logging('Was stopped with learning rate: ' +  str( corresponding_learning_rates[loading_model_index] ) )
    #         logging('Loading file : ' + model_filenames[loading_model_index])
    #         ori_model.load_state_dict(torch.load(model_filenames[loading_model_index]))



    model = nn.DataParallel(ori_model)
    # if the learning rate was not loaded then we set it equal to the initial one
    # if lr == 0:
    #     lr = config.init_lr


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

    for epoch in range(1000):
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

            sentence_scores = Variable(data['sentence_scores'])

            # get model's output
            # predict_support = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=False)

            # calculating loss
            # loss_2 = nll_average(predict_support.view(-1, 2), is_support.view(-1) )

            # logging('Batch training loss :' + str(loss_2.item()) )

            # loss = loss_2


            logit1, logit2, predict_type, predict_support = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, sentence_scores, return_yp=False)
            loss_1 = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0)
            loss_2 = nll_average(predict_support.view(-1, 2), is_support.view(-1).long())


            loss = loss_1 + config.sp_lambda * loss_2

            # log both losses :

            # with open(os.path.join(config.save, 'losses_log.txt'), 'a+') as f_log:
            #     s = 'Answer loss :' + str(loss_1.data.item()) + ', SF loss :' + str(loss_2.data.item()) + ', Total loss :' + str(loss.data.item())
            #     f_log.write(s + '\n')




            # update train metrics
            # train_metrics = update_sp(train_metrics, predict_support, is_support)


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
                logging('| epoch {:3d} | step {:6d} | ms/batch {:5.2f} | train loss {:8.3f}'.format(epoch, global_step, elapsed*1000/config.period, cur_loss))
                # logging('| epoch {:3d} | step {:6d} | ms/batch {:5.2f} | train loss {:8.3f} | SP EM {:8.3f} | SP f1 {:8.3f} | SP Prec {:8.3f} | SP Recall {:8.3f}'.format(epoch, global_step, elapsed*1000/config.period, cur_loss, train_metrics['sp_em'], train_metrics['sp_f1'], train_metrics['sp_prec'], train_metrics['sp_recall']))

                total_loss = 0
                # train_metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}
                start_time = time.time()


                # logging('Saving model...')
                torch.save(ori_model.state_dict(), os.path.join( 'temp_model') )
                torch.save(optimizer.state_dict(), os.path.join( 'temp_optimizer') )

            # if global_step % config.checkpoint == 0:
            #     model.eval()

            #     # metrics = evaluate_batch(build_dev_iterator(), model, 5, dev_eval_file, config)
            #     eval_metrics = evaluate_batch(build_dev_iterator(), model, 0, dev_eval_file, config)
            #     model.train()

            #     logging('-' * 89)
            #     # logging('| eval {:6d} in epoch {:3d} | time: {:5.2f}s | dev loss {:8.3f} | EM {:.4f} | F1 {:.4f}'.format(global_step//config.checkpoint,
            #     #     epoch, time.time()-eval_start_time, metrics['loss'], metrics['exact_match'], metrics['f1']))
            #     logging('| eval {:6d} in epoch {:3d} | time: {:5.2f}s | dev loss {:8.3f}| SP EM {:8.3f} | SP f1 {:8.3f} | SP Prec {:8.3f} | SP Recall {:8.3f}'.format(global_step//config.checkpoint,
            #         epoch, time.time()-eval_start_time, eval_metrics['loss'], eval_metrics['sp_em'], eval_metrics['sp_f1'], eval_metrics['sp_prec'], eval_metrics['sp_recall']))
            #     logging('-' * 89)

            #     eval_start_time = time.time()

            if global_step % config.checkpoint == 0:
                model.eval()
                metrics = evaluate_batch(build_dev_iterator(), model, 0, dev_eval_file, config)
                model.train()

                logging('-' * 89)
                logging('| eval {:6d} in epoch {:3d} | time: {:5.2f}s | dev loss {:8.3f} | EM {:.4f} | F1 {:.4f}'.format(global_step//config.checkpoint,
                    epoch, time.time()-eval_start_time, metrics['loss'], metrics['exact_match'], metrics['f1']))
                logging('-' * 89)

                eval_start_time = time.time()


                dev_F1 = metrics['f1']
                if best_dev_F1 is None or dev_F1 > best_dev_F1:
                    best_dev_F1 = dev_F1
                    torch.save(ori_model.state_dict(), os.path.join(config.save, 'model.pt'))
                    torch.save(optimizer.state_dict(), os.path.join(config.save,  'optimizer.pt') )
                    cur_patience = 0
                else:
                    cur_patience += 1
                    if cur_patience >= config.patience:
                        stop_train = True
                        break

                # if eval_metrics['sp_f1'] > best_dev_sp_f1:
                #     best_dev_sp_f1 = eval_metrics['sp_f1']
                #     torch.save(ori_model.state_dict(), os.path.join(config.save, 'model.pt'))
                #     torch.save(optimizer.state_dict(), os.path.join(config.save,  'optimizer.pt') )
                #     # cur_lr_decrease_patience = 0
                #     cur_patience = 0
                # else:
                #     cur_patience += 1
                #     if cur_patience >= config.patience:
                #         stop_train = True
                #         break

                        # lr *= 0.75
                        # for param_group in optimizer.param_groups:
                        #     param_group['lr'] = lr
                        # if lr < config.init_lr * 1e-2:
                        #     stop_train = True
                        #     break
                        # cur_patience = 0
                        
                    # cur_lr_decrease_patience += 1
                    # if cur_lr_decrease_patience >= config.lr_decrease_patience:
                    #     lr *= 0.75
                    #     cur_early_stop_patience +=1
                    #     if cur_early_stop_patience >= config.early_stop_patience:
                    #         stop_train = True
                    #         break

                    #     for param_group in optimizer.param_groups:
                    #         param_group['lr'] = lr


        if stop_train: break
    logging('best_dev_F1 {}'.format(best_dev_F1))

    # delete last temporary trained model, since the training has completed
    print('Deleting last temp model files...')
    # for dp, dn, filenames in os.walk('.'):
    #     for ff in filenames:
    # if ff.endswith("last_model.pt"):
    os.remove('temp_model')
    os.remove('temp_optimizer')



def evaluate_batch(data_source, model, max_batches, eval_file, config):
    answer_dict = {}
    sp_dict = {}
    total_loss, step_cnt = 0, 0
    iter = data_source
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

        sentence_scores = Variable(data['sentence_scores'])


        # print('Test 0')

        logit1, logit2, predict_type, predict_support, yp1, yp2  = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, sentence_scores, return_yp=True)
        
        # print('Test 4')


        # logit1, logit2, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
        loss = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0) + config.sp_lambda * nll_average(predict_support.view(-1, 2), is_support.view(-1).long())
        
        # print('Test 5')
        
        answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        total_loss += loss.data.item()
        step_cnt += 1
    loss = total_loss / step_cnt
    metrics = evaluate(eval_file, answer_dict)
    metrics['loss'] = loss

    return metrics


def predict(data_source, model, eval_file, config, prediction_file):
    answer_dict = {}
    sp_dict = {}
    sp_th = config.sp_threshold
    for step, data in enumerate(tqdm(data_source)):
        context_idxs = Variable(data['context_idxs'], volatile=True)
        ques_idxs = Variable(data['ques_idxs'], volatile=True)
        context_char_idxs = Variable(data['context_char_idxs'], volatile=True)
        ques_char_idxs = Variable(data['ques_char_idxs'], volatile=True)
        context_lens = Variable(data['context_lens'], volatile=True)
        start_mapping = Variable(data['start_mapping'], volatile=True)
        end_mapping = Variable(data['end_mapping'], volatile=True)
        all_mapping = Variable(data['all_mapping'], volatile=True)

        sentence_scores = Variable(data['sentence_scores'])


        logit1, logit2, predict_type, predict_support, yp1, yp2  = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, sentence_scores, return_yp=True)
        

        # logit1, logit2, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
        answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(predict_support[:, :, 1]).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = data['ids'][i]
            for j in range(predict_support_np.shape[1]):
                if j >= len(eval_file[cur_id]['sent2title_ids']): break
                if predict_support_np[i, j] > sp_th:
                    cur_sp_pred.append(eval_file[cur_id]['sent2title_ids'][j])
            sp_dict.update({cur_id: cur_sp_pred})

    prediction = {'answer': answer_dict, 'sp': sp_dict}
    with open(prediction_file, 'w') as f:
        json.dump(prediction, f)


# def update_sp(metrics, predict_support, is_support):

#     # def logging(s, print_=True, log_=True):
#     #     if print_:
#     #         print(s)
#     #     if log_:
#     #         with open(os.path.join('predictions_log.txt'), 'a+') as f_log:
#     #             f_log.write(s + '\n')

                
#     setneces_per_sample = is_support.size()[1]
#     bsz = is_support.size()[0]

#     predict_support = torch.nn.functional.softmax(predict_support, dim=2)

#     batch_prec = 0
#     batch_recall = 0
#     batch_f1 = 0
#     batch_em = 0
    
#     for b in range(bsz):
#         tp, fp, fn = 0, 0, 0


#         for i in range (setneces_per_sample):
#             # if this sentences was created for batch padding purposes then we move to the next sample
#             if is_support[b,i] == -100:
#                 continue


#             # print('Prediction :', predict_support[i,1].item(), ', Ground truth :',  is_support[i].item())
#             predicted = predict_support[b,i,1] >= predict_support[b,i,0]
#             ground_truth = (is_support[b,i] == 1)
#             # print(predicted.item(), ground_truth.item())

#             # logging('ground_truth :' + str(ground_truth.data.item()) + ', predicted :' + str(predicted.data.item()) +' Predicted outputs : (Not SP) ' + str(predict_support[b,i,0].data.item()) + ' , (SP) ' + str(predict_support[b,i,1].data.item()))
            
#             if predicted:
#                 # true positive
#                 if ground_truth:
#                     tp += 1
#                 # false positive
#                 else:
#                     fp += 1
#             else:
#                 # false negative
#                 if ground_truth:
#                     fn +=1
#             # print(tp , fp, fn)

#         # calculate evaluation metrics per sample

#         prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
#         recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
#         f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
#         em = 1.0 if fp + fn == 0 else 0.0
#         # add them in order to calculate the batch average
#         batch_prec += prec
#         batch_recall += recall
#         batch_f1 += f1
#         batch_em += em


#     # adding the average evaluations of this step to the total evaluations
#     metrics['sp_em'] += batch_em / bsz
#     metrics['sp_f1'] += batch_f1 / bsz
#     metrics['sp_prec'] += batch_prec / bsz
#     metrics['sp_recall'] += batch_recall / bsz
#     return metrics

# def evaluate_batch(data_source, model, max_batches, eval_file, config):
#     answer_dict = {}
#     sp_dict = {}
#     total_loss, step_cnt = 0, 0
#     iter = data_source
#     eval_metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}

#     for step, data in enumerate(iter):
#         if step >= max_batches and max_batches > 0: break

#         context_idxs = Variable(data['context_idxs'], volatile=True)
#         ques_idxs = Variable(data['ques_idxs'], volatile=True)
#         context_char_idxs = Variable(data['context_char_idxs'], volatile=True)
#         ques_char_idxs = Variable(data['ques_char_idxs'], volatile=True)
#         context_lens = Variable(data['context_lens'], volatile=True)
#         y1 = Variable(data['y1'], volatile=True)
#         y2 = Variable(data['y2'], volatile=True)
#         q_type = Variable(data['q_type'], volatile=True)
#         is_support = Variable(data['is_support'], volatile=True)
#         start_mapping = Variable(data['start_mapping'], volatile=True)
#         end_mapping = Variable(data['end_mapping'], volatile=True)
#         all_mapping = Variable(data['all_mapping'], volatile=True)


#         predict_support= model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
        
#         # update eval metrics
#         eval_metrics = update_sp(eval_metrics, predict_support, is_support)

#         # logit1, logit2, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
#         # loss = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0) + config.sp_lambda * nll_average(predict_support.view(-1, 2), is_support.view(-1))

#         loss =  nll_average(predict_support.view(-1, 2), is_support.view(-1))
        
#         # answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
#         # answer_dict.update(answer_dict_)

#         total_loss += loss.data.item()
#         step_cnt += 1


#     # avegage metrics
#     for key in eval_metrics:
#         eval_metrics[key] /= float(step_cnt)

#     eval_metrics['loss'] = total_loss

#     # return metrics
#     return eval_metrics

# def predict(data_source, model, eval_file, config, prediction_file):
#     answer_dict = {}
#     sp_dict = {}
#     sp_th = config.sp_threshold


#     eval_metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}

#     for step, data in enumerate(tqdm(data_source)):
#         context_idxs = Variable(data['context_idxs'], volatile=True)
#         ques_idxs = Variable(data['ques_idxs'], volatile=True)
#         context_char_idxs = Variable(data['context_char_idxs'], volatile=True)
#         ques_char_idxs = Variable(data['ques_char_idxs'], volatile=True)
#         context_lens = Variable(data['context_lens'], volatile=True)
#         start_mapping = Variable(data['start_mapping'], volatile=True)
#         end_mapping = Variable(data['end_mapping'], volatile=True)
#         all_mapping = Variable(data['all_mapping'], volatile=True)

#         # logit1, logit2, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
#         # answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
#         # answer_dict.update(answer_dict_)

#         predict_support = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)

#         # eval_metrics = update_sp(eval_metrics, predict_support, is_support)


#         predict_support_np = torch.sigmoid(predict_support[:, :, 1]).data.cpu().numpy()
#         for i in range(predict_support_np.shape[0]):
#             cur_sp_pred = []
#             cur_id = data['ids'][i]
#             for j in range(predict_support_np.shape[1]):
#                 if j >= len(eval_file[cur_id]['sent2title_ids']): break
#                 if predict_support_np[i, j] > sp_th:
#                     cur_sp_pred.append(eval_file[cur_id]['sent2title_ids'][j])
#             sp_dict.update({cur_id: cur_sp_pred})

#     # prediction = {'answer': answer_dict, 'sp': sp_dict}
#     prediction = {'sp': sp_dict}
#     with open(prediction_file, 'w') as f:
#         json.dump(prediction, f)

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
    ori_model = model.cuda()
    # ori_model = model
    ori_model.load_state_dict(torch.load(os.path.join(config.save, 'model.pt')))
    # model = ori_model
    model = nn.DataParallel(ori_model)

    model.eval()
    predict(build_dev_iterator(), model, dev_eval_file, config, config.prediction_file)

