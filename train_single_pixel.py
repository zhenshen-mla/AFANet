import os
import numpy as np
from data_process.Dataloader import make_data_loader
from models.pixel_single import DeepLab
from utils.loss_func.depth import loss_huber
from utils.loss_func.semantic import SegmentationLosses
from utils.scheduler import LR_Scheduler
from tensorboardX import SummaryWriter
import torch
from utils.metrics.evaluator import Evaluator
from utils.metrics.error import depth_threshold, depth_error

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

writer = SummaryWriter(comment='_STL')


def main():
    # args
    nclass1 = 40
    nclass2 = 1
    lr = 0.01
    num_epochs = 400
    batch_size = 12
    low_rms = 100.0
    low_rel = 100.0
    high_delta1 = 0.0
    high_delta2 = 0.0
    high_delta3 = 0.0
    high_miou = 0.0
    high_pacc = 0.0
    drop_last = True
    path = '/weights/'

    # Define Dataloader
    train_loader, val_loader = make_data_loader(batch_size=batch_size, drop_last=drop_last)
    # Define network
    model = DeepLab(num_classes1=nclass1,
                    num_classes2=nclass2,
                    output_stride=16,
                    freeze_bn=False)

    train_params = [{'params': model.get_1x_lr_params(), 'lr': lr},
                    {'params': model.get_10x_lr_params(), 'lr': lr * 10}]

    # Define Optimizer
    optimizer = torch.optim.SGD(train_params, momentum=0.9, weight_decay=5e-4, nesterov=False)

    # Define Criterion
    criterion1 = SegmentationLosses(weight=None, cuda=True).build_loss(mode='ce')
    criterion2 = loss_huber()

    # Define Evaluator
    evaluator = Evaluator(nclass1)

    # Define lr scheduler
    scheduler = LR_Scheduler(mode='poly', base_lr=lr, num_epochs=300, iters_per_epoch=len(train_loader))
    model = model.cuda()

    T = 2.0
    avg_cost = np.zeros([num_epochs, 2], dtype=np.float32)
    lambda_weight = np.ones([2, num_epochs])

    for epoch in range(num_epochs):
        train_loss1 = 0.0
        train_loss2 = 0.0
        train_loss3 = 0.0
        count = 0

        index = epoch
        if index == 0 or index == 1:
            lambda_weight[:, index] = 1.0
        else:
            w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
            w_2 = avg_cost[index - 1, 1] / avg_cost[index - 2, 1]
            lambda_weight[0, index] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
            lambda_weight[1, index] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))

        model.train()
        for i, sample in enumerate(train_loader):
            image, target1, target2 = sample['image'], sample['label'], sample['depth']
            image, target1, target2 = image.cuda(), target1.cuda(), target2.cuda()
            scheduler(optimizer, i, epoch, high_miou, low_rel)
            optimizer.zero_grad()
            output1, output2 = model(image)
            loss1 = criterion1(output1, target1)
            loss2 = criterion2(output2, target2)

            train_loss = [loss1, loss2]
            loss = torch.mean(sum(lambda_weight[j, index] * train_loss[j] for j in range(2)))
            loss.backward()
            optimizer.step()
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss.item()
            count += 1
        train_loss1 = train_loss1 / count
        train_loss2 = train_loss2 / count
        train_loss3 = train_loss3 / count
        avg_cost[index, :] = [train_loss1, train_loss2]

        print(
            'Training  [Epoch: %d, mIoU: %.4f, PAcc: %.4f, delta1: %.4f, delta2: %.4f, delta3: %.4f, rms: %.4f, rel: %4f, numImages: %5d, Loss1: %.3f, Loss2: %.3f, Loss: %.3f]' % (
            epoch, high_miou, high_pacc, high_delta1, high_delta2, high_delta3, low_rms, low_rel,
            i * batch_size + image.data.shape[0], train_loss1, train_loss2, train_loss3))
        writer.add_scalar('scalar/loss_seg_train', train_loss1, epoch)
        writer.add_scalar('scalar/loss_depth_train', train_loss2, epoch)
        writer.add_scalar('scalar/loss_multi_train', train_loss3, epoch)

        if epoch < 200:
            continue
        elif epoch >= 200 and epoch % 2 == 0:
            continue

        model.eval()
        evaluator.reset()
        val_loss1 = 0.0
        val_loss2 = 0.0
        count = 0
        num_samples = len(val_loader)

        abs_err_sum = 0.0
        rel_err_sum = 0.0
        rel_sqr_sum = 0.0
        log_10_sum = 0.0
        RMSE_linear_sum = 0.0
        RMSE_loge_sum = 0.0
        RMSE_log10_sum = 0.0
        Threshold_1_25_sum = 0.0
        Threshold_1_25_2_sum = 0.0
        Threshold_1_25_3_sum = 0.0

        for i, sample in enumerate(val_loader):
            input_var, target1, gt_var = sample['image'].cuda(), sample['label'].cuda(), sample['depth'].cuda()
            with torch.no_grad():
                output1, output2 = model(input_var)
            loss1 = criterion1(output1, target1)
            loss2 = criterion2(output2, gt_var)
            val_loss1 += loss1.item()
            val_loss2 += loss2.item()
            count += 1
            ### semantic
            pred1 = output1.data.cpu().numpy()
            target1 = target1.cpu().numpy()
            pred1 = np.argmax(pred1, axis=1)
            # Add batch sample into evaluator
            evaluator.add_batch(target1, pred1)
            ### depth
            gt_var = gt_var.unsqueeze(dim=1)
            abs_err, rel_err, rel_sqr, log_10, RMSE_linear, RMSE_loge, RMSE_log10 = depth_error(output2, gt_var)

            abs_err_sum += abs_err
            rel_err_sum += rel_err
            rel_sqr_sum += rel_sqr
            log_10_sum += log_10
            RMSE_linear_sum += RMSE_linear
            RMSE_loge_sum += RMSE_loge
            RMSE_log10_sum += RMSE_log10

            Threshold_1_25, Threshold_1_25_2, Threshold_1_25_3 = depth_threshold(output2, gt_var)
            Threshold_1_25_sum += Threshold_1_25
            Threshold_1_25_2_sum += Threshold_1_25_2
            Threshold_1_25_3_sum += Threshold_1_25_3

        ### semantic
        val_loss1 = val_loss1 / num_samples
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Validation  [Epoch: %d, numImages: %5d, Loss: %.3f]' % (
        epoch, i * batch_size + image.data.shape[0], val_loss1))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        writer.add_scalar('scalar/loss_seg_val', val_loss1, epoch)
        writer.add_scalar('scalar/Acc_val', Acc, epoch)
        writer.add_scalar('scalar/Acc_class_val', Acc_class, epoch)
        writer.add_scalar('scalar/mIou_val', mIoU, epoch)
        writer.add_scalar('scalar/FWIou_val', FWIoU, epoch)

        if mIoU > high_miou:
            high_miou = mIoU
            torch.save(model.state_dict(), path+'Single_seg.pkl')
            print('save the seg ' + str(epoch) + 'model, replace the previous model')
        if Acc > high_pacc:
            high_pacc = Acc

        ### depth
        val_loss2 = val_loss2 / num_samples
        print('Validation  [Loss: %.3f]' % val_loss2)

        abs_err_sum /= num_samples
        rel_err_sum /= num_samples
        rel_sqr_sum /= num_samples
        log_10_sum /= num_samples
        RMSE_linear_sum /= num_samples
        RMSE_loge_sum /= num_samples
        RMSE_log10_sum /= num_samples
        Threshold_1_25_sum /= num_samples
        Threshold_1_25_2_sum /= num_samples
        Threshold_1_25_3_sum /= num_samples

        print('abs_err: {}'.format(abs_err_sum))
        print('rel_err: {}'.format(rel_err_sum))
        print('rel_sqr: {}'.format(rel_sqr_sum))
        print('log_10: {}'.format(log_10_sum))
        print('RMSE_linear: {}'.format(RMSE_linear_sum))
        print('RMSE_loge: {}'.format(RMSE_loge_sum))
        print('RMSE_log10: {}'.format(RMSE_log10_sum))
        print('Threshold_1_25: {}'.format(Threshold_1_25_sum))
        print('Threshold_1_25_2: {}'.format(Threshold_1_25_2_sum))
        print('Threshold_1_25_3: {}'.format(Threshold_1_25_3_sum))

        writer.add_scalar('scalar/loss_depth_val', val_loss2, epoch)
        writer.add_scalar('scalar/abs_err', abs_err_sum, epoch)
        writer.add_scalar('scalar/rel_err', rel_err_sum, epoch)
        writer.add_scalar('scalar/rel_sqr', rel_sqr_sum, epoch)
        writer.add_scalar('scalar/log_10', log_10_sum, epoch)
        writer.add_scalar('scalar/RMSE_linear', RMSE_linear_sum, epoch)
        writer.add_scalar('scalar/RMSE_loge', RMSE_loge_sum, epoch)
        writer.add_scalar('scalar/RMSE_log10', RMSE_log10_sum, epoch)
        writer.add_scalar('scalar/Threshold_1_25', Threshold_1_25_sum, epoch)
        writer.add_scalar('scalar/Threshold_1_25_2', Threshold_1_25_2_sum, epoch)
        writer.add_scalar('scalar/Threshold_1_25_3', Threshold_1_25_3_sum, epoch)

        if rel_err_sum < low_rel:
            low_rel = rel_err_sum
            torch.save(model.state_dict(), path + 'Single_depth.pkl')
            print('save the depth ' + str(epoch) + 'model, replace the previous model')
        if RMSE_linear_sum < low_rms:
            low_rms = RMSE_linear_sum
        if Threshold_1_25_sum > high_delta1:
            high_delta1 = Threshold_1_25_sum
        if Threshold_1_25_2_sum > high_delta2:
            high_delta2 = Threshold_1_25_2_sum
        if Threshold_1_25_3_sum > high_delta3:
            high_delta3 = Threshold_1_25_3_sum


if __name__ == "__main__":
    main()
    writer.close()



