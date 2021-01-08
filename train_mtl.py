import torch.nn as nn
import os
import torch
from functools import reduce
import copy
from tqdm import tqdm_notebook as tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import get_bb
import skimage.transform as skTrans
from scipy import stats
from PIL import Image
from scipy.special import softmax
# from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

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
          classes=['wildtype', 'oligo', 'mutant'],
          pad=0,
          weight_dir = '../model_weights/',
          weight_outfile_prefix='temp',
          pretrained=None):


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


        # writer.add_scalar('training loss',
        #             running_loss / num_samples_processed,
        #             epoch * len(dataloaders[phase]) + i)
        writer.add_scalar('training loss',
                    running_loss / num_samples_processed,
                    epoch * len(dataloaders[phase]) + i)
        writer.add_scalar('training seg loss',
                    running_seg_loss / num_samples_processed,
                    epoch * len(dataloaders[phase]) + i)
        writer.add_scalar('training cls loss',
                    running_cls_loss / num_samples_processed,
                    epoch * len(dataloaders[phase]) + i)
        # writer.add_scalar('training surv loss',
        #             running_surv_loss / num_samples_processed,
        #             epoch * len(dataloaders[phase]) + i)
        #
        # writer.add_scalar('training dice wt',
        #             dice_wt,
        #             epoch * len(dataloaders[phase]) + i)
        # writer.add_scalar('training dice core',
        #             dice_core,
        #             epoch * len(dataloaders[phase]) + i)
        # writer.add_scalar('training dice enh',
        #             dice_enh,
        #             epoch * len(dataloaders[phase]) + i)


        writer.add_scalar('training acc',
                            training_class_acc,
                            epoch * len(dataloaders[phase]) + i)
        writer.add_scalar('training AUC',
                            training_auc_score,
                            epoch * len(dataloaders[phase]) + i)
        writer.add_scalar('training average AUC-AUC',
                            np.mean([training_auc_score, training_class_acc]),
                            epoch * len(dataloaders[phase]) + i)

        # writer.add_scalar('training f1',
        #                     f1,
        #                     epoch * len(dataloaders[phase]) + i)
        # writer.add_scalar('training precision',
        #                     precision,
        #                     epoch * len(dataloaders[phase]) + i)
        # writer.add_scalar('training recall',
        #                     recall,
        #                     epoch * len(dataloaders[phase]) + i)




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

                # labels = torch.tensor(labels, dtype=torch.long, device=device)
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

        if verbose:
            print('----- validation metrics ------')
        valid_slice_class_acc, valid_slice_class_AUC, dice_wt, dice_core, dice_enh, f1, precision, recall = class_accuracies(pred_tensor = valid_train['slice_preds'],
                                              label_tensor = valid_train['labels'],
                                              probs_list = valid_train['slice_probs'],
                                              dice_dict=valid_dice_dict,
                                              verbose = verbose,
                                              classes=classes,
                                              phase = 'no preds given')
        auc_acc_mean = np.mean([valid_slice_class_acc, valid_slice_class_AUC])
        # if epoch % 2 == 0:
        #     print('  >> val_loss', val_loss, 'epoch', epoch)
        #     print('  >> val AUC ', valid_slice_class_AUC, '| mean acc auc', auc_acc_mean, '| acc',  valid_slice_class_acc, '| epoch', epoch)



        # if verbose:
        #     print('\nLoss')
        #     print(" - Loss:\t", val_loss)
        #     print(" - Seg Loss:\t", running_seg_loss / num_samples_processed)
        #     print(" - Cls Loss:\t", running_cls_loss / num_samples_processed)
        #     print(" - Surv Loss:\t", running_surv_loss / num_samples_processed)
        #     print('----- end validation ------\n')
        #
        #     print('-- END OF EPOCH', epoch , '--')

        writer.add_scalar('val acc',
                            valid_slice_class_acc,
                            epoch * len(dataloaders[phase]) + i)
        writer.add_scalar('val AUC',
                            valid_slice_class_AUC,
                            epoch * len(dataloaders[phase]) + i)

        writer.add_scalar('val average AUC-AUC',
                            auc_acc_mean,
                            epoch * len(dataloaders[phase]) + i)

        writer.add_scalar('val f1',
                            f1,
                            epoch * len(dataloaders[phase]) + i)
        writer.add_scalar('val precision',
                            precision,
                            epoch * len(dataloaders[phase]) + i)
        writer.add_scalar('val recall',
                            recall,
                            epoch * len(dataloaders[phase]) + i)

        # writer.add_scalar('val dice wt',
        #             dice_wt,
        #             epoch * len(dataloaders[phase]) + i)
        # writer.add_scalar('val dice core',
        #             dice_core,
        #             epoch * len(dataloaders[phase]) + i)
        # writer.add_scalar('val dice enh',
        #             dice_enh,
        #             epoch * len(dataloaders[phase]) + i)

        if auc_acc_mean > best_naive_auc_acc_mean:
            best_naive_auc_acc_mean = auc_acc_mean
            print('New Best AUC-acc average:\t', best_naive_auc_acc_mean, '\tin epoch', epoch)
            auc_acc_str = str(np.round(best_naive_auc_acc_mean, 4)).split('.')[-1]
            auc_str = str(np.round(valid_slice_class_AUC, 4)).split('.')[-1]
            acc_str = str(np.round(valid_slice_class_acc, 4)).split('.')[-1]

            outfile_suffix = '_AUC-acc-mean_' + auc_acc_str + '_auc-' + auc_str + '_acc-' + acc_str +  '_epoch_' + str(epoch) + '.pth'
            auc_acc_mean_dir = weight_dir + 'auc_acc_mean1/'

            if os.path.isfile(naive_auc_outfile):
                os.remove(naive_auc_outfile)
            if not os.path.exists(auc_acc_mean_dir):
                os.makedirs(auc_acc_mean_dir)
            naive_auc_outfile = auc_acc_mean_dir + weight_outfile_prefix + outfile_suffix
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': running_loss
                        }, naive_auc_outfile)
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

        if dice_wt > best_wt_dice_score:
            best_wt_dice_score = dice_wt
            print('New Best Dice:\t', best_wt_dice_score, '\tin epoch', epoch)
            outfile_suffix = '_dice_' + str(np.round(best_wt_dice_score, 4)).split('.')[-1] + '_epoch_' + str(epoch) +  '.pth'
            dice_dir = weight_dir + 'dice1/'

            if os.path.isfile(naive_dice_outfile):
                os.remove(naive_dice_outfile)
            if not os.path.exists(dice_dir):
                os.makedirs(dice_dir)
            naive_dice_outfile = dice_dir + weight_outfile_prefix + outfile_suffix
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': running_loss
                        }, naive_dice_outfile)

        if valid_slice_class_acc > best_naive_acc:
            best_naive_acc = valid_slice_class_acc
            print('New Best ACC:\t', best_naive_acc, '\tin epoch', epoch)
            # best_model_wts = copy.deepcopy(model.state_dict())
            outfile_suffix = '_ACC_' + str(np.round(best_naive_acc, 4)).split('.')[-1] + '_epoch_' + str(epoch) +  '.pth'
            acc_dir = weight_dir + 'acc1/'


            if not os.path.exists(acc_dir):
                os.makedirs(acc_dir)
            if os.path.isfile(naive_acc_outfile):
                os.remove(naive_acc_outfile)
            naive_acc_outfile = acc_dir + weight_outfile_prefix + outfile_suffix
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': running_loss
                        }, naive_acc_outfile)




    print('Finished Training')
    return model, best_model_wts, best_naive_auc, best_naive_acc, best_naive_auc_acc_mean
