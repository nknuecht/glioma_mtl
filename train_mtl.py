import torch.nn as nn
import os
import torch
from functools import reduce
import copy
from tqdm.notebook import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import get_bb
import skimage.transform as skTrans
from scipy import stats
from PIL import Image
from scipy.special import softmax

from models.mtl_model import GBMNetMTL
import pickle
import torch.optim as optim

from utils import get_bb_3D_torch, class_accuracies, get_dice_scores


def train(model,
          dataloaders,
          data_transforms,
          optimizer,
          scheduler,
          dataset_sizes,
          writer,
          num_epochs=25,
          verbose=False,
          device='cpu',
          channels=4,
          classes=['wildtype', 'mutant'],
          weight_dir = '../model_weights/',
          weight_outfile_prefix='temp',
          pretrained=None):

    '''
    This function trains an MTL model and saves its best weights
    Arguments
    ---------
    model: model to train
    dataloaders: dictonary of training and validation dataloaders
    data_transforms: dictonary of training and validation data transforms
    optimizer: optimizer (Adam)
    scheduler: training schedule
    dataset_sizes: dictonary of the size of the training and validation datasets
    writer: tensorboard write
    verbose: enable print statements
    device: cpu or gpu
    channels: MR channels (1 or 4)
    classes: names of classification classes
    weight_dir: directory to save best model weights to
    weight_outfile_prefix: prefix of files saved from this model
    pretrained: whether pretrained model is passed

    Outputs
    ---------
    model: trained model
    best_model_wts: model weights for model with best AUC score
    best_naive_auc: best AUC score
    '''

    best_model_wts = copy.deepcopy(model.state_dict())
    naive_acc_outfile, naive_auc_outfile, vol_acc_outfile, vol_auc_outfile, naive_dice_outfile = '', '', '', '', ''

    best_naive_auc, best_naive_acc, best_naive_auc_acc_mean = 0.0, 0.0, 0
    best_vol_acc, best_vol_auc, best_wt_dice_score = 0.0, 0.0, 0.0

    val_loss = torch.from_numpy(np.asarray([np.inf]))
    for epoch in tqdm(range(num_epochs)):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        epoch_pred_list, epoch_label_list = [], []
        train_dict = {'preds':[], 'labels':[], 'probs':[]}
        train_dice_dict = {'dice_wt':[], 'dice_core':[], 'dice_enh':[]}
        phase = 'train'
        model.train()

        running_seg_loss, running_cls_loss, running_surv_loss, running_loss, running_corrects = 0.0, 0.0, 0.0, 0.0, 0

        # Iterate over data.
        num_samples_processed = 0
        for i, data in enumerate(dataloaders[phase]):

            (inputs, seg_image, genomic_data, seg_probs), labels, (OS, event), bratsID  = data
            inputs, labels = inputs.to(device), labels.to(device)
            OS, event = OS.to(device), event.to(device)
            seg_image, seg_probs, genomic_data = seg_image.to(device), seg_probs.to(device), genomic_data.to(device)

            seg_image = seg_image.squeeze(1)
            seg_image = seg_image.type(torch.int64)
            seg_probs = seg_probs.squeeze(1)
            seg_probs = seg_probs.type(torch.int64)

            num_samples_processed += inputs.shape[0]


            seg_object = model(image_data = inputs,  # input
                        genome_data = genomic_data,
                        seg_probs=seg_probs,
                        seg_gt=seg_image,
                        class_labels=labels,
                        compute_loss=True,
                        event=event,
                        OS=OS,
                        bratsIDs=bratsID)

            # mem = torch.cuda.memory_allocated(device=device)

            class_pred = seg_object.class_out # the fusion network's output
            seg_pred = seg_object.seg_out # predicted segmentation mask (why do we need to output this?)
            seg_loss = seg_object.seg_loss
            class_loss = seg_object.class_loss
            surv_loss = seg_object.surv_loss
            surv_risk = seg_object.surv_risk
            ci = seg_object.ci


            # get_dice_scores
            seg_pred = seg_pred.max(1)[1].data.byte().cpu().numpy()
            dice_wt, dice_core, dice_enh = get_dice_scores(im1=seg_pred, im2=seg_image)

            train_dice_dict['dice_wt'].append(dice_wt)
            train_dice_dict['dice_core'].append(dice_core)
            train_dice_dict['dice_enh'].append(dice_enh)


            loss = seg_loss + class_loss + surv_loss

            output_probs = softmax(class_pred.detach().cpu().numpy(), axis=1)[:, 1]
            _, preds = torch.max(class_pred, 1)

            # labels = torch.tensor(labels, dtype=torch.long, device=device)
            labels = labels.detach().clone()
            # loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # running_loss += loss.item() * inputs.size(0)
            running_seg_loss += seg_loss.item()
            running_cls_loss += class_loss.item()
            running_surv_loss += surv_loss.item()
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

            train_dict['preds'] = train_dict['preds'] + list(preds)
            train_dict['labels'] = train_dict['labels'] + list(labels.data)
            train_dict['probs'] = train_dict['probs'] + list(output_probs)


        idx = [i for i, x in enumerate(train_dict['labels']) if x == 255]
        for index in sorted(idx, reverse=True):
            del train_dict['preds'][index]
            del train_dict['probs'][index]
            del train_dict['labels'][index]

        ## training metrics
        training_class_acc, training_auc_score, dice_wt, dice_core, dice_enh, f1, precision, recall = class_accuracies(pred_tensor = train_dict['preds'],
                                              label_tensor = train_dict['labels'],
                                              probs_list = train_dict['probs'],
                                              dice_dict=train_dice_dict,
                                              verbose = verbose,
                                              classes=classes,
                                              phase='train')



        writer.add_scalar('training loss',
                    running_loss / num_samples_processed,
                    epoch * len(dataloaders[phase]) + i)
        writer.add_scalar('training seg loss',
                    running_seg_loss / num_samples_processed,
                    epoch * len(dataloaders[phase]) + i)
        writer.add_scalar('training cls loss',
                    running_cls_loss / num_samples_processed,
                    epoch * len(dataloaders[phase]) + i)



        writer.add_scalar('training acc',
                            training_class_acc,
                            epoch * len(dataloaders[phase]) + i)
        writer.add_scalar('training AUC',
                            training_auc_score,
                            epoch * len(dataloaders[phase]) + i)






        with torch.no_grad():
            ###### validate #############
            phase = 'val'
            valid_train = {'slice_preds':[], 'slice_probs':[], 'volume_preds':[], 'volume_probs':[], 'labels':[]}
            valid_dice_dict = {'dice_wt':[], 'dice_core':[], 'dice_enh':[]}
            model.eval()

            running_seg_loss, running_cls_loss, running_surv_loss, running_loss, running_corrects = 0, 0, 0, 0.0, 0

            num_samples_processed = 0
            for i, data in enumerate(dataloaders[phase]):

                (inputs, seg_image, genomic_data, seg_probs), labels, (OS, event), bratsID  = data
                output_shape = seg_image.shape
                inputs, labels = inputs.to(device), labels.to(device)
                OS, event = OS.to(device), event.to(device)
                seg_image, seg_probs, genomic_data = seg_image.to(device), seg_probs.to(device), genomic_data.to(device)
                seg_image = seg_image.squeeze(1)
                seg_image = seg_image.type(torch.int64)

                seg_probs = seg_probs.squeeze(1)
                seg_probs = seg_probs.type(torch.int64)


                num_samples_processed += inputs.shape[0]

                seg_object = model(image_data = inputs,  # input
                        genome_data = genomic_data,
                        seg_probs=seg_probs,
                        seg_gt=seg_image,
                        class_labels=labels,
                        compute_loss=True, # should change this to False sometime
                        event=event,
                        OS=OS,
                        bratsIDs=bratsID)

                class_pred = seg_object.class_out # the fusion network's output
                seg_pred = seg_object.seg_out # predicted segmentation mask (why do we need to output this?)
                seg_loss = seg_object.seg_loss
                class_loss = seg_object.class_loss
                surv_loss = seg_object.surv_loss
                surv_risk = seg_object.surv_risk
                ci = seg_object.ci

                seg_pred = seg_pred.max(1)[1].data.byte().cpu().numpy()
                dice_wt, dice_core, dice_enh = get_dice_scores(im1=seg_pred, im2=seg_image)

                valid_dice_dict['dice_wt'].append(dice_wt)
                valid_dice_dict['dice_core'].append(dice_core)
                valid_dice_dict['dice_enh'].append(dice_enh)

                output_probs = softmax(class_pred.detach().cpu().numpy(), axis=1)[:, 1]
                _, slice_preds = torch.max(class_pred, 1) ## what do we do with the preds?

                labels = labels.detach().clone()


                loss = seg_loss + class_loss + surv_loss


                running_seg_loss += seg_loss.item()
                running_cls_loss += class_loss.item()
                running_surv_loss += surv_loss.item()
                running_loss += loss.item()
                running_corrects += torch.sum(slice_preds == labels.data)


                valid_train['slice_preds'] = valid_train['slice_preds'] + list(slice_preds)
                valid_train['slice_probs'] = valid_train['slice_probs'] + list(output_probs)
                valid_train['labels'] = valid_train['labels'] + list(labels.data)

                writer.add_scalar('val loss',
                            running_loss / num_samples_processed,
                            epoch * len(dataloaders[phase]) + i)
                writer.add_scalar('val seg loss',
                            running_seg_loss / num_samples_processed,
                            epoch * len(dataloaders[phase]) + i)
                writer.add_scalar('val cls loss',
                            running_cls_loss / num_samples_processed,
                            epoch * len(dataloaders[phase]) + i)
                writer.add_scalar('val surv loss',
                            running_surv_loss / num_samples_processed,
                            epoch * len(dataloaders[phase]) + i)

        val_loss = running_loss / num_samples_processed
        scheduler.step(val_loss)


        valid_slice_class_acc, valid_slice_class_AUC, dice_wt, dice_core, dice_enh, f1, precision, recall = class_accuracies(pred_tensor = valid_train['slice_preds'],
                                              label_tensor = valid_train['labels'],
                                              probs_list = valid_train['slice_probs'],
                                              dice_dict=valid_dice_dict,
                                              verbose = verbose,
                                              classes=classes,
                                              phase = 'no preds given')
        auc_acc_mean = np.mean([valid_slice_class_acc, valid_slice_class_AUC])


        writer.add_scalar('val acc',
                            valid_slice_class_acc,
                            epoch * len(dataloaders[phase]) + i)
        writer.add_scalar('val AUC',
                            valid_slice_class_AUC,
                            epoch * len(dataloaders[phase]) + i)



        if epoch > 5:
            if valid_slice_class_AUC > best_naive_auc:
                best_naive_auc = valid_slice_class_AUC
                print('New Best AUC:\t', best_naive_auc, '\tin epoch', epoch)

                best_model_wts = copy.deepcopy(model.state_dict()) # when we return the best weights, we mean best AUC weights

                outfile_suffix = '_AUC_' + str(np.round(best_naive_auc, 4)).split('.')[-1] + '_epoch_' + str(epoch) + '.pth'
                auc_dir = weight_dir + 'auc1/'

                if os.path.isfile(naive_auc_outfile):
                    os.remove(naive_auc_outfile)
                if not os.path.exists(auc_dir):
                    os.makedirs(auc_dir)
                naive_auc_outfile = auc_dir + weight_outfile_prefix + outfile_suffix
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': running_loss
                            }, naive_auc_outfile)



    print('Finished Training')
    return model, best_model_wts, best_naive_auc




def mtl_experiment(dataloaders,
                   data_transforms,
                   dataset_sizes,
                   best_model_loc,
                   best_auc_list,
                   best_acc_list,
                   weight_outfile_prefix,
                   channels,
                   loss_weights,
                   seg_4class_weights,
                   seg_2class_weights,
                   seg_loss_weight,
                   surv_loss_weight,
                   device,
                   brats_seg_ids,
                   writer,
                   class_names,
                   model_weights_dir='../model_weights/results/',
                   epochs=50,
                   iterations=10,
                   standard_unlabled_loss=True,
                   include_genomic_data=True,
                   modality=None,
                   take_surv_loss=False,
                   g_in_features=50,
                   g_out_features=128):

    '''
    This function trains an MTL model and saves its best weights
    Arguments
    ---------
    dataloaders: dictonary of training and validation dataloaders
    data_transforms: dictonary of training and validation data transforms
    dataset_sizes: dictonary of the size of the training and validation datasets
    best_model_loc: location of 3D-ESPNet's best weights (downloaded from 3D-ESPNet's repo)
    best_auc_list: running auc scores over experiment
    best_acc_list: running average accuracy scores over experiment
    weight_outfile_prefix: prefix of files saved from this model
    channels: MR channels (1 or 4)
    seg_4class_weights: segmentation class weights for 4-class segmenation masks
    seg_2class_weights: segmentation class weights for 2-class segmenation masks
    seg_loss_weight: weight of segmentation pentaly in network loss function
    surv_loss_weight: weight of survival pentaly in network loss function (not currently implemented)
    device: cpu or gpu
    brats_seg_ids: list of samples that have ground truth segmentation masks
    writer: tensorboard write
    class_names: names of classes to classify (currently 2)
    model_weights_dir: directory to save best model weights to
    epochs: number epochs
    iterations: iteration experiment is on
    standard_unlabled_loss: whether to take 2-class segmentation loss with before or after softmax
    include_genomic_data: include SCNA input data
    modality: for 1-channel input, indicate which MR modality serves as input
    take_surv_loss: bool: add survival loss to network loss function (not currently implemented)
    seg_classes: number of segmentation class (4 for ground truth segmentation masks; 2 otherwise)
    g_in_features: size of SCNA input
    g_out_features: size of hidden layer in SCNA input branch

    Outputs
    ---------
    best_auc_list: running auc scores over experiment
    best_acc_list: running average accuracy scores over experiment
    '''

    # mtl model
    gbm_net = GBMNetMTL(g_in_features=g_in_features,
                        g_out_features=g_out_features,
                        n_classes=len(class_names),
                        n_volumes=channels,
                        pretrained=best_model_loc,
                        class_loss_weights = loss_weights,
                        seg_4class_weights=seg_4class_weights,
                        seg_2class_weights=seg_2class_weights,
                        seg_loss_scale=seg_loss_weight,
                        surv_loss_scale=surv_loss_weight,
                        device = device,
                        brats_seg_ids=brats_seg_ids,
                        standard_unlabled_loss=standard_unlabled_loss,
                        fusion_net_flag=include_genomic_data,
                        modality=modality,
                        take_surv_loss=take_surv_loss)
    gbm_net = gbm_net.to(device)

    optimizer_gbmnet = optim.Adam(gbm_net.parameters(), lr=0.0005) # change to adami

    exp_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_gbmnet,
                                                         mode='min',
                                                         factor=0.1,
                                                         patience=10, # number of epochs with no change
                                                         verbose=True,
                                                         threshold=0.0001,
                                                         threshold_mode='rel',
                                                         cooldown=0,
                                                         min_lr=0,
                                                         eps=1e-08)


    # train mtl model
    model, best_wts, best_auc = train(model=gbm_net,
                   dataloaders=dataloaders,
                   data_transforms=data_transforms,
                   optimizer=optimizer_gbmnet,
                   scheduler=exp_scheduler,
                   writer=writer,
                   num_epochs=epochs,
                   verbose=False,
                   device=device,
                   dataset_sizes=dataset_sizes,
                   channels=channels,
                   classes=class_names,
                   weight_outfile_prefix=weight_outfile_prefix)

    del gbm_net
    del model

    # add epoch best score
    best_auc_list.append(best_auc)

    # make model weight directory
    if not os.path.exists(model_weights_dir):
        os.makedirs(model_weights_dir)

    # save best model weights of this epoch
    results_outfile_dir = weight_outfile_prefix + '_epochs-' + str(epochs) +'_iterations-' + str(iterations)
    with open('../model_weights/results/auc_' + results_outfile_dir + '.txt', "wb") as fp:   #Pickling
        pickle.dump(best_auc_list, fp)

    return best_auc_list, best_acc_list
