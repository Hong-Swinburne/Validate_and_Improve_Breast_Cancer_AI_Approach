import os
import torch
import numpy as np
from torch import mode
from tqdm import tqdm
from src.utilities.metric import compute_metric

val_best_roc_auc = 0.

def running_one_epoch(epoch, output_path, model, train_loader, val_loader, criterion, optimizer, device, threshold, beta, train_loss, train_acc, \
    valid_loss, valid_acc, writer):
    model.train()
    epoch_loss = 0
    prediction = np.empty(shape=[0, 2], dtype=np.int)
    N_train = len(train_loader)

    for step, (imgs, labels) in tqdm(enumerate(train_loader)):
        imgs, labels = imgs.to(device), labels.to(device)
        y_global, y_local, y_fusion, saliency_map = model(imgs)
        
        y_global = y_global[:,1:]
        y_local = y_local[:,1:]
        y_fusion = y_fusion[:,1:]
        saliency_map = saliency_map[:,1:,:,:]

        loss1 = criterion(y_global, labels)
        loss2 = criterion(y_local, labels)
        loss3 = criterion(y_fusion, labels)
        loss4 = beta * saliency_map.mean()
        loss =  loss1 + loss2 + loss3 + loss4

        epoch_loss += loss.item()
        result = np.concatenate([y_fusion.cpu().data.numpy(), labels.cpu().data.numpy()], axis=1)
        prediction = np.concatenate([prediction, result], axis=0)
        print('epoch {0:d} --- {1:.4f} --- loss: {2:.6f}'.format(epoch+1, step / N_train, loss.item()), end='\r', flush=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss /= step
    TP, FN, TN, FP, acc, roc_auc = compute_metric(prediction, threshold)
    sen = TP/(TP+FN)
    spe = TN/(TN+FP)
    lr = optimizer.param_groups[0]["lr"]
    print('{} Epoch finished ! Train Loss: {}, lr: {}, train acc:{}, train roc_auc:{}'.format(epoch+1, epoch_loss, format(lr,'.2e'), acc, roc_auc))
    train_loss.append(epoch_loss)
    train_acc.append(acc)
    val_loss, val_acc, val_roc_auc, val_TP, val_FN, val_TN, val_FP = val_epoch(epoch, output_path, model, val_loader, criterion, device, threshold, \
        beta, valid_loss, valid_acc)
    val_sen = val_TP/(val_TP+val_FN)
    val_spe = val_TN/(val_TN+val_FP)
    writer.writerow({'epoch':epoch+1, 'train_loss':epoch_loss, 'train_acc':acc, 'train_auc':roc_auc, 'train_sen':sen, 'train_spe':spe, 'val_loss':val_loss,\
        'val_acc':val_acc, 'val_auc':val_roc_auc, 'val_sen':val_sen, 'val_spe':val_spe})
    
    return lr

@torch.no_grad()
def val_epoch(epoch, output_path, model, loader, criterion, device, threshold, beta, valid_loss, valid_acc):
    global val_best_roc_auc
    model.eval()
    epoch_loss = 0
    prediction = np.empty(shape=[0, 2], dtype=np.int)
    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        y_global, y_local, y_fusion, saliency_map = model(imgs)
        
        y_global = y_global[:,1:]
        y_local = y_local[:,1:]
        y_fusion = y_fusion[:,1:]
        saliency_map = saliency_map[:,1:,:,:]

        loss1 = criterion(y_global, labels)
        loss2 = criterion(y_local, labels)
        loss3 = criterion(y_fusion, labels)
        loss4 = beta * saliency_map.mean()
        loss =  loss1 + loss2 + loss3 + loss4
        epoch_loss += loss.item()

        result = np.concatenate([y_fusion.cpu().data.numpy(), labels.cpu().data.numpy()], axis=1)
        prediction = np.concatenate([prediction, result], axis=0)

    epoch_loss /= step
    print('==> ### validate metric ###')
    print('Epoch: {} --- Val Loss: {}'.format(epoch+1, epoch_loss))
    print('loss_global: {0:.4f} --- loss_local: {1:.4f} --- loss_fusion: {2:.4f}'.format(loss1, loss2, loss3))

    # compute metric
    total = len(loader.dataset)
    TP, FN, TN, FP, acc, roc_auc = compute_metric(prediction, threshold)
    print('threshold: %.2f --- TP: %d --- FN: %d --- TN: %d --- FP: %d'%(threshold, TP, FN, TN, FP))
    print('acc: %f --- roc_auc: %f'%(acc, roc_auc))
    print('Total: %d --- cur_best_val_roc_auc: %f'%(total, val_best_roc_auc))
    valid_loss.append(epoch_loss)
    valid_acc.append(acc)

    # save best
    if roc_auc > val_best_roc_auc:
        val_best_roc_auc = roc_auc
        save_epoch(epoch, output_path, model)
        print('Save New Best Successfully at epoch %d, valid_acc %.2f, valid_auc %.2f'%(epoch+1, acc, roc_auc))

    print('')
    return epoch_loss, acc, roc_auc, TP, FN, TN, FP

def save_epoch(epoch, output_path, model, filename=None):
    if not filename:
        path = os.path.join(output_path, 'val_best_model.pth')
    else:
        path = os.path.join(output_path, filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, path)