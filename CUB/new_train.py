import pdb
import os
import sys
import argparse, yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import numpy as np
from analysis import Logger, AverageMeter, accuracy, binary_accuracy
from CUB.new_iterative_norm import IterNorm

from CUB.dataset import load_data, find_class_imbalance, find_partition_indices_by_IG, find_partition_indices_byRandom
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE
from CUB.models import ModelXtoCY, ModelXtoChat_ChatToY, ModelXtoY, ModelXtoC, ModelOracleCtoY, ModelXtoCtoY
from CUB.auxiliary_loss import calculate_ortho_loss

def run_epoch(epoch, model, optimizer, loader, loss_meter, acc_meter, criterion, attr_criterion, args, mi_args,
                is_training, indices, mi_optimizer = None, train_estimate_mi_meter = None, train_mi_learning_loss_meter = None, train_crossC_loss_meter = None):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """
    if is_training:
        model.train()
    else:
        model.eval()

    for _, data in enumerate(loader):
        if attr_criterion is None:
            inputs, labels = data
            attr_labels, attr_labels_var = None, None
        else:
            inputs, labels, attr_labels = data
            if args.n_attributes < 112:
                attr_labels = [attr_labels[index] for index in indices]
            if args.n_attributes > 1:
                attr_labels = [i.long() for i in attr_labels]
                attr_labels = torch.stack(attr_labels).t()#.float() #N x 312
            else:
                if isinstance(attr_labels, list):
                    attr_labels = attr_labels[0]
                attr_labels = attr_labels.unsqueeze(1)
            attr_labels_var = torch.autograd.Variable(attr_labels).float()
            attr_labels_var = attr_labels_var.cuda() if torch.cuda.is_available() else attr_labels_var

        inputs_var = torch.autograd.Variable(inputs)
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels)
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var

        mi_estimator_loss = 0.
        estimate_mi = 0.
        crossC = 0.
        
        if args.disentangle == "IterNorm" and not args.semi_supervise:
            outputs, aux_outputs = model(inputs_var)
            losses = []
            out_start = 0
            if not args.bottleneck: #loss main is for the main task label (always the first output)
                loss_main = 1.0 * criterion(outputs[0], labels_var) + 0.4 * criterion(aux_outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    losses.append(args.attr_loss_weight * (1.0 * attr_criterion[i](outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i]) \
                                                            + 0.4 * attr_criterion[i](aux_outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i])))
        elif args.semi_supervise and is_training and args.use_aux:
            stage1_outs = model.first_model(inputs_var)
            for i, stage1_out in enumerate(stage1_outs):
                attr_outputs = stage1_out
                stage2_inputs = attr_outputs
                concepts = torch.cat(stage2_inputs[:model.n_attributes], dim=1)
                if model.residue > 0:
                    residue = torch.cat(stage2_inputs[model.n_attributes:], dim=1)

                if args.disentangle == "IterNorm":
                        # print("perform IterNorm ***********")
                        ItN = IterNorm(args.n_attributes + args.residue, num_channels = args.n_attributes + args.residue, dim=2, T=7, momentum=1, affine=True)
                        ItN.train()
                        iter_output = ItN(torch.cat([concepts.detach(), residue], dim=1))
                        stage1_out = torch.split(iter_output, 1, dim = 1)
                        concepts = torch.cat(stage1_out[:model.n_attributes], dim=1)
                        residue = torch.cat(stage1_out[model.n_attributes:], dim=1)

                if model.should_detach:
                    if model.residue > 0:
                        if args.semi_supervise:
                            stage2_inputs = torch.cat([attr_labels_var, residue], dim=1) ## execute this for semi_supervise learning
                            # print("training time intervention ****************************")
                        else:
                            stage2_inputs = torch.cat([concepts.detach(), residue], dim=1)
                    else:
                        if args.semi_supervise:
                            # stage2_inputs = torch.cat([attr_labels_var], dim=1)
                            stage2_inputs = attr_labels_var
                            # print("run baseline for semi-supversied training (independent)")
                        else:
                            stage2_inputs = torch.cat(stage2_inputs, dim=1).detach()
                else: 
                    if model.residue > 0:
                        if args.semi_supervise:
                            stage2_inputs = torch.cat([attr_labels_var, residue], dim=1)
                        else:
                            stage2_inputs = torch.cat([concepts, residue], dim=1)
                    else:
                        if args.semi_supervise:
                            # stage2_inputs = torch.cat(attr_labels_var, dim=1)
                            stage2_inputs = attr_labels_var
                        else:
                            stage2_inputs = torch.cat(stage2_inputs, dim=1)
                
                if i == 0:  
                    if args.disentangle == "crossC":
                        # print("compute crossC loss")
                        crossC = calculate_ortho_loss(concepts, residue)
                        train_crossC_loss_meter.update(crossC.data.cpu().numpy(), inputs.size(0))

                    if mi_optimizer is not None and epoch > model.mi_args.start_epoch:
                        # print("estimate mi")
                        mi_optimizer.zero_grad()
                        mi_estimator_loss = model.mi_estimator.estimator_loss(concepts, residue)
                        mi_estimator_loss.backward()
                        mi_optimizer.step()

                        if model.should_detach:
                            estimate_mi = model.mi_estimator(concepts.detach(), residue)
                        else:
                            estimate_mi = model.mi_estimator(concepts, residue)
                        train_estimate_mi_meter.update(mi_args.weight * estimate_mi.data.cpu().numpy(), inputs.size(0))
                        train_mi_learning_loss_meter.update(mi_estimator_loss.data.cpu().numpy(), inputs.size(0))
                    
                    outputs = [model.sec_model(stage2_inputs)]
                    outputs.extend(stage1_out)

                if i == 1:
                    aux_outputs = [model.sec_model(stage2_inputs)]
                    aux_outputs.extend(stage1_out)

            losses = []
            out_start = 0
            if not args.bottleneck: #loss main is for the main task label (always the first output)
                loss_main = 1.0 * criterion(outputs[0], labels_var) + 0.4 * criterion(aux_outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    
                    losses.append(args.attr_loss_weight * (1.0 * attr_criterion[i](outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i]) \
                                                            + 0.4 * attr_criterion[i](aux_outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i])))
                if args.disentangle == "crossC":
                    losses.append(crossC)
                if mi_optimizer is not None and epoch > model.mi_args.start_epoch:
                    losses.append(mi_args.weight * estimate_mi)
        else: #testing or no aux logits
            outputs = model(inputs_var)
            losses = []
            out_start = 0
            if not args.bottleneck:
                loss_main = criterion(outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    losses.append(args.attr_loss_weight * attr_criterion[i](outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i]))

        if args.bottleneck: #attribute accuracy
            outputs = outputs[:args.n_attributes]
            sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
            acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
        else:
            acc = accuracy(outputs[0], labels, topk=(1,)) #only care about class prediction accuracy
            acc_meter.update(acc[0], inputs.size(0))

        if attr_criterion is not None:
            if (mi_optimizer is not None and epoch > model.mi_args.start_epoch) or (args.disentangle == "crossC"):
                total_loss = losses[0] + sum(losses[1:-1])
                if args.normalize_loss:
                    total_loss = total_loss / (1 + args.attr_loss_weight * args.n_attributes)
                total_loss += losses[-1]                
            elif args.bottleneck:
                total_loss = sum(losses)/ args.n_attributes
            else: #cotraining, loss by class prediction and loss by attribute prediction have the same weight
                total_loss = losses[0] + sum(losses[1:])
                if args.normalize_loss:
                    total_loss = total_loss / (1 + args.attr_loss_weight * args.n_attributes)
        else: #finetune
            total_loss = sum(losses)
        loss_meter.update(total_loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return loss_meter, acc_meter, train_estimate_mi_meter, train_mi_learning_loss_meter, train_crossC_loss_meter

def train(model, args, mi_args):
    # Determine imbalance
    os.chdir("/home/shelton/supervised-concept")
    imbalance = None
    if args.use_attr and not args.no_img and args.weighted_loss:
        train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        if args.weighted_loss == 'multiple':
            imbalance = find_class_imbalance(train_data_path, True)
        else:
            imbalance = find_class_imbalance(train_data_path, False)

    if not os.path.exists(args.log_dir): # job restarted by cluster
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, f'log_{args.seed}.txt'))
    logger.write(str(args) + '\n')
    logger.write(str(imbalance) + '\n')
    logger.flush()

    if args.reduce == 'ig':
        indices = find_partition_indices_by_IG(args.n_attributes, args)
    elif args.reduce == 'random':
        indices = find_partition_indices_byRandom(args.n_attributes, args)

    logger.write(str(indices) + '\n')
    logger.flush()

    logger.write(f"{torch.cuda.device_count()} GPUs in current env" + "\n")
    logger.flush()
    logger.write(f"use GPU is {torch.cuda.is_available()}\n")
    logger.flush()
    logger.write(f"residue size: {args.residue}\n")
    logger.flush()


    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    if args.use_attr and not args.no_img:
        attr_criterion = [] #separate criterion (loss function) for each attribute
        if args.weighted_loss:
            assert(imbalance is not None)
            for index, ratio in enumerate(imbalance):
                if args.n_attributes < 112:
                    if index in indices:
                        attr_criterion.append(torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda()))
                else:
                    attr_criterion.append(torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda()))
        else:
            for i in range(args.n_attributes):
                attr_criterion.append(torch.nn.CrossEntropyLoss())
    else:
        attr_criterion = None

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    mi_optimizer = None
    if args.disentangle == "club":
        #mi_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.mi_estimator.parameters()), lr=0.001)
        mi_optimizer = torch.optim.Adam(model.mi_estimator.get_parameters(), lr=mi_args.lr, betas=(0.5, 0.999))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    stop_epoch = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step

    train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    logger.write('train data path: %s\n' % train_data_path)

    if args.ckpt: #retraining
        train_loader = load_data([train_data_path, val_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                                 n_class_attr=args.n_class_attr, resampling=args.resampling)
        # train_loader = load_data([train_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                                #  n_class_attr=args.n_class_attr, resampling=args.resampling)
        val_loader = None
    else:
        train_loader = load_data([train_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                                 n_class_attr=args.n_class_attr, resampling=args.resampling)
        val_loader = load_data([val_data_path], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir, n_class_attr=args.n_class_attr)

    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(0, args.epochs):
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        train_estimate_mi_meter = AverageMeter()
        train_mi_learning_loss_meter = AverageMeter()
        train_crossC_loss_meter = AverageMeter()

        if args.no_img:
            train_loss_meter, train_acc_meter = run_epoch_simple(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, args, is_training=True, indices = indices)
        else: 
            train_loss_meter, train_acc_meter, train_estimate_mi_meter, train_mi_learning_loss_meter, train_crossC_loss_meter = run_epoch(epoch, model, optimizer, train_loader, train_loss_meter,
            train_acc_meter, criterion, attr_criterion, args, mi_args, is_training=True, indices = indices, mi_optimizer = mi_optimizer,
            train_estimate_mi_meter = train_estimate_mi_meter, train_mi_learning_loss_meter = train_mi_learning_loss_meter, train_crossC_loss_meter = train_crossC_loss_meter)

        if not args.ckpt: # evaluate on val set
            print(not args.ckpt)
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()
            train_mi_loss_meter = AverageMeter()
            train_mi_learning_loss_meter = AverageMeter()
        
            with torch.no_grad():
                if args.no_img:
                    val_loss_meter, val_acc_meter = run_epoch_simple(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, args, is_training=False, indices = indices)
                else:
                    val_loss_meter, val_acc_meter, _, _, _ = run_epoch(epoch, model, optimizer, val_loader, val_loss_meter,
            val_acc_meter, criterion, attr_criterion, args, mi_args, is_training=False, indices = indices, mi_optimizer = mi_optimizer,
            train_estimate_mi_meter = train_estimate_mi_meter, train_mi_learning_loss_meter = train_mi_learning_loss_meter, train_crossC_loss_meter = train_crossC_loss_meter)
        else: #retraining
            val_loss_meter = train_loss_meter
            val_acc_meter = train_acc_meter

        # if best_val_acc < val_acc_meter.avg:
        #     best_val_epoch = epoch
        #     best_val_acc = val_acc_meter.avg
        #     logger.write('New model best model at epoch %d\n' % epoch)
        #     torch.save(model, os.path.join(args.log_dir, 'best_model_%d.pt' % (args.seed)))

        if best_val_loss > val_loss_meter.avg:
            best_val_epoch = epoch
            best_val_loss = val_loss_meter.avg
            logger.write('New model best model at epoch %d\n' % epoch)
            torch.save(model, os.path.join(args.log_dir, 'best_model_%d.pt' % (args.seed)))


        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg
        logger.write('Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\t'
                    'Val loss: %.4f\tVal acc: %.4f\tMi: %.4f\tMi learning loss: %.4f\tCross Correlation: %.4f\t'
                    'Best val epoch: %d\n'
                    % (epoch, train_loss_avg, train_acc_meter.avg, val_loss_avg, val_acc_meter.avg, train_estimate_mi_meter.avg, 
                    train_mi_learning_loss_meter.avg, train_crossC_loss_meter.avg,best_val_epoch)) 
        logger.flush()
        
        if epoch <= stop_epoch:
            scheduler.step(epoch) #scheduler step to update lr at the end of epoch     
        #inspect lr
        if epoch % 10 == 0:
            print('Current lr:', scheduler.get_lr())

        # if epoch % args.save_step == 0:
        #     torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 20:
            print("Early stopping because acc hasn't improved for a long time")
            break
        # break #fixeme-------------------------------------

def train_X_to_C_to_y(args, mi_args):
    model = ModelXtoCtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, freeze=args.freeze,
                         num_classes=N_CLASSES, use_aux=args.use_aux, n_attributes=args.n_attributes,
                         expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid, residue=args.residue,
                         residue_indept = args.residue_indept, should_detach = args.should_detach, disentangle = args.disentangle, mi_args = mi_args)
    train(model, args, mi_args)

def parse_arguments(experiment):
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='CUB Training')
    parser.add_argument('dataset', type=str, help='Name of the dataset.')
    parser.add_argument('exp', type=str,
                        choices=['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                                 'Standard', 'Multitask', 'Joint', 'Probe',
                                 'TTI', 'Robustness', 'HyperparameterSearch'],
                        help='Name of experiment to run.')

    if experiment == 'Probe':
        return (probe.parse_arguments(parser),)

    elif experiment == 'TTI':
        return (tti.parse_arguments(parser),)

    elif experiment == 'Robustness':
        return (gen_cub_synthetic.parse_arguments(parser),)

    elif experiment == 'HyperparameterSearch':
        return (hyperopt.parse_arguments(parser),)

    else:
        parser.add_argument('-log_dir', default=None, help='where the trained model is saved')
        parser.add_argument('-batch_size', '-b', type=int, help='mini-batch size')
        parser.add_argument('-seed', default=1, type=int, help='Numpy and torch seed.')
        parser.add_argument('-epochs', '-e', type=int, help='epochs for training process')
        parser.add_argument('-save_step', default=1000, type=int, help='number of epochs to save model')
        parser.add_argument('-lr', type=float, help="learning rate")
        parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
        parser.add_argument('-pretrained', '-p', action='store_true',
                            help='whether to load pretrained model & just fine-tune')
        parser.add_argument('-freeze', action='store_true', help='whether to freeze the bottom part of inception network')
        parser.add_argument('-use_aux', action='store_true', help='whether to use aux logits')
        parser.add_argument('-use_attr', action='store_true',
                            help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)')
        parser.add_argument('-attr_loss_weight', default=1.0, type=float, help='weight for loss by predicting attributes')
        parser.add_argument('-no_img', action='store_true',
                            help='if included, only use attributes (and not raw imgs) for class prediction')
        parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
        parser.add_argument('-weighted_loss', default='', # note: may need to reduce lr
                            help='Whether to use weighted loss for single attribute or multiple ones')
        parser.add_argument('-uncertain_labels', action='store_true',
                            help='whether to use (normalized) attribute certainties as labels')
        parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES,
                            help='whether to apply bottlenecks to only a few attributes')
        parser.add_argument('-expand_dim', type=int, default=0,
                            help='dimension of hidden layer (if we want to increase model capacity) - for bottleneck only')
        parser.add_argument('-n_class_attr', type=int, default=2,
                            help='whether attr prediction is a binary or triary classification')
        parser.add_argument('-data_dir', default='official_datasets', help='directory to the training data')
        parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
        parser.add_argument('-resampling', help='Whether to use resampling', action='store_true')
        parser.add_argument('-end2end', action='store_true',
                            help='Whether to train X -> A -> Y end to end. Train cmd is the same as cotraining + this arg')
        parser.add_argument('-optimizer', default='SGD', help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
        parser.add_argument('-ckpt', default='', help='For retraining on both train + val set')
        parser.add_argument('-scheduler_step', type=int, default=1000,
                            help='Number of steps before decaying current learning rate by half')
        parser.add_argument('-normalize_loss', action='store_true',
                            help='Whether to normalize loss by taking attr_loss_weight into account')
        parser.add_argument('-use_relu', action='store_true',
                            help='Whether to include relu activation before using attributes to predict Y. '
                                 'For end2end & bottleneck model')
        parser.add_argument('-use_sigmoid', action='store_true',
                            help='Whether to include sigmoid activation before using attributes to predict Y. '
                                 'For end2end & bottleneck model')
        parser.add_argument('-connect_CY', action='store_true',
                            help='Whether to use concepts as auxiliary features (in multitasking) to predict Y')
        parser.add_argument('-residue', type = int, default = 0, help = 'The size of neurons in the residue layer')
        parser.add_argument('-residue_indept', type = int, default = 0, help = 'The size of last layer in independent residue model')
        parser.add_argument('-subset', type = int, default = 112, help = 'Percentage of subset of concepts used')
        parser.add_argument('-ray', action='store_true', help = 'whether use ray to execute multiple experiment')
        parser.add_argument('-ray_tune', action='store_true', help = 'whether use ray to tune the hyperparameters')
        parser.add_argument('-should_detach', action='store_true', help = 'whether to detach concept layer')
        parser.add_argument('-disentangle', default=None, help='which disentangle method is used')
        parser.add_argument('-model_dir', default=None, help='pretained model directory for mi estimator')
        parser.add_argument('-reduce', default=None, help='whether to remove the size of concept randomly or by feature importance')
        parser.add_argument('-semi_supervise', action='store_true', help = 'whether to replace concept with true value during training')

        args = parser.parse_args()
        args.n_attributes = args.subset
        args.three_class = (args.n_class_attr == 3)
        return args
