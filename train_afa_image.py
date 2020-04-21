import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from data_process.Dataloader import make_loader
from models.image_AFA import make_network
from tensorboardX import SummaryWriter
from utils.scheduler import LR_Scheduler
from sklearn import metrics


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
writer = SummaryWriter(comment='AFA_Image')


def load_pretrained_model(model, path):

    path = path + 'Single.pkl'
    pretrain_dict = torch.load(path)
    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        if k[2:4] != 'fc':
            if k in state_dict:
                print(k)
                model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    return model


def main():
    path = '/weights/'

    best_pred_age = 0.0
    best_pred_gender = 0.0
    best_acc_age = 0.0
    best_acc_gender = 0.0
    best_micro_age = 0.0
    best_micro_gender = 0.0
    best_macro_age = 0.0
    best_macro_gender = 0.0
    lr = 0.001
    num_epochs = 50

    print('\nloading the dataset ...\n')
    train_data, val_data, trainloader, valloader = make_loader()
    print(len(train_data), len(val_data), len(trainloader), len(valloader))
    print('done')

    ## load network
    print('\nloading the network ...\n')
    model = make_network()

    model = load_pretrained_model(model, path)
    criterion_gender = nn.CrossEntropyLoss()
    criterion_age = nn.CrossEntropyLoss()
    print('done')

    ## move to GPU
    print('\nmoving to GPU ...\n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion_gender.to(device)
    criterion_age.to(device)
    print('done')

    ### optimizer
    train_params = [{'params': model.get_1x_lr_params(), 'lr': lr},
                    {'params': model.get_10x_lr_params(), 'lr': lr * 10}]

    optimizer = optim.SGD(train_params, momentum=0.9, weight_decay=5e-4, nesterov=False)
    scheduler = LR_Scheduler(mode='step', base_lr=lr, num_epochs=num_epochs, iters_per_epoch=len(trainloader), lr_step=10)

    # training
    print('\nstart training ...\n')

    T = 2.0
    avg_cost = np.zeros([num_epochs, 2], dtype=np.float32)
    lambda_weight = np.ones([2, num_epochs])

    for epoch in range(num_epochs):

        index = epoch
        if index == 0 or index == 1:
            lambda_weight[:, index] = 1.0
        else:
            w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
            w_2 = avg_cost[index - 1, 1] / avg_cost[index - 2, 1]
            lambda_weight[0, index] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
            lambda_weight[1, index] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))

        running_loss_gender = 0.0
        running_correct_gender = 0
        running_total_gender = 0

        running_loss_age = 0.0
        running_correct_age = 0
        running_total_age = 0

        acc_gender = 0.0
        acc_age = 0.0
        micro_gender = 0.0
        micro_age = 0.0
        macro_gender = 0.0
        macro_age = 0.0
        count = 0

        model.train()
        for batch_idx, (data, target_gender, target_age) in enumerate(trainloader):
            data, target_gender, target_age = data.to(device), target_gender.to(device), target_age.to(device)
            scheduler(optimizer, batch_idx, epoch, best_pred_gender, best_pred_age)
            optimizer.zero_grad()

            # forward
            pred_gender, pred_age = model(data)
            # backward
            loss_gender = criterion_gender(pred_gender, target_gender)
            loss_age = criterion_age(pred_age, target_age)

            train_loss = [loss_gender, loss_age]
            loss = torch.mean(sum(lambda_weight[j, index] * train_loss[j] for j in range(2)))

            loss.backward()
            optimizer.step()

            predict_gender = torch.argmax(pred_gender, 1)
            predict_age = torch.argmax(pred_age, 1)
            a1 = metrics.accuracy_score(target_gender.cpu(), predict_gender.cpu())
            b1 = metrics.f1_score(target_gender.cpu(), predict_gender.cpu(), average='micro')
            c1 = metrics.f1_score(target_gender.cpu(), predict_gender.cpu(), average='macro')
            a2 = metrics.accuracy_score(target_age.cpu(), predict_age.cpu())
            b2 = metrics.f1_score(target_age.cpu(), predict_age.cpu(), average='micro')
            c2 = metrics.f1_score(target_age.cpu(), predict_age.cpu(), average='macro')

            acc_gender += a1
            acc_age += a2
            micro_gender += b1
            micro_age += b2
            macro_gender += c1
            macro_age += c2
            count += 1

            correct_gender = torch.eq(predict_gender, target_gender).sum().double().item()
            correct_age = torch.eq(predict_age, target_age).sum().double().item()

            running_loss_gender += loss_gender.item()
            running_loss_age += loss_age.item()
            running_correct_gender += correct_gender
            running_correct_age += correct_age
            running_total_gender += target_gender.size(0)
            running_total_age += target_age.size(0)

        loss_gender = running_loss_gender * 32 / running_total_gender
        accuracy_gender = 100 * running_correct_gender / running_total_gender
        loss_age = running_loss_age * 32 / running_total_age
        accuracy_age = 100 * running_correct_age / running_total_age

        acc_gender /= count
        acc_age /= count
        micro_gender /= count
        micro_age /= count
        macro_gender /= count
        macro_age /= count

        avg_cost[index, :] = [loss_gender, loss_age]

        writer.add_scalar('scalar/loss_gender_train', loss_gender, epoch)
        writer.add_scalar('scalar/loss_age_train', loss_age, epoch)
        writer.add_scalar('scalar/accuracy_gender_train', accuracy_gender, epoch)
        writer.add_scalar('scalar/accuracy_age_train', accuracy_age, epoch)
        writer.add_scalar('scalar/acc_gender_train', acc_gender, epoch)
        writer.add_scalar('scalar/acc_age_train', acc_age, epoch)
        writer.add_scalar('scalar/micro_gender_train', micro_gender, epoch)
        writer.add_scalar('scalar/micro_age_train', micro_age, epoch)
        writer.add_scalar('scalar/macro_gender_train', macro_gender, epoch)
        writer.add_scalar('scalar/macro_age_train', macro_age, epoch)

        print('previous best ',
              '    Epoch[%d /100], best_pred_gender=%.4f %%, best_pred_age = %.4f %%, best_acc_gender = %.4f, best_acc_age = %.4f, best_micro_gender = %.4f, best_micro_age = %.4f,best_macro_gender = %.4f, best_macro_age = %.4f' %
              (epoch + 1, best_pred_gender, best_pred_age, best_acc_gender, best_acc_age, best_micro_gender, best_micro_age, best_macro_gender, best_macro_age))

        model.eval()
        with torch.no_grad():
            running_loss_gender = 0.0
            running_correct_gender = 0
            running_total_gender = 0

            running_loss_age = 0.0
            running_correct_age = 0
            running_total_age = 0

            acc_gender = 0.0
            acc_age = 0.0
            micro_gender = 0.0
            micro_age = 0.0
            macro_gender = 0.0
            macro_age = 0.0
            count = 0

            for batch_idx, (data, target_gender, target_age) in enumerate(valloader):
                data, target_gender, target_age = data.to(device), target_gender.to(device), target_age.to(device)
                optimizer.zero_grad()
                # forward
                pred_gender, pred_age = model(data)
                # backward
                loss_gender = criterion_gender(pred_gender, target_gender)
                loss_age = criterion_age(pred_age, target_age)

                predict_gender = torch.argmax(pred_gender, 1)
                predict_age = torch.argmax(pred_age, 1)

                a1 = metrics.accuracy_score(target_gender.cpu(), predict_gender.cpu())
                b1 = metrics.f1_score(target_gender.cpu(), predict_gender.cpu(), average='micro')
                c1 = metrics.f1_score(target_gender.cpu(), predict_gender.cpu(), average='macro')
                # print('(2) ', a1, b1, c1)

                a2 = metrics.accuracy_score(target_age.cpu(), predict_age.cpu())
                b2 = metrics.f1_score(target_age.cpu(), predict_age.cpu(), average='micro')
                c2 = metrics.f1_score(target_age.cpu(), predict_age.cpu(), average='macro')
                # print('(4)', a2, b2, c2)

                correct_gender = torch.eq(predict_gender, target_gender).sum().double().item()
                correct_age = torch.eq(predict_age, target_age).sum().double().item()

                running_loss_gender += loss_gender.item()
                running_loss_age += loss_age.item()
                running_correct_gender += correct_gender
                running_correct_age += correct_age
                running_total_gender += target_gender.size(0)
                running_total_age += target_age.size(0)

                acc_gender += a1
                acc_age += a2
                micro_gender += b1
                micro_age += b2
                macro_gender += c1
                macro_age += c2
                count += 1

            loss_gender = running_loss_gender * 32 / running_total_gender
            accuracy_gender = 100 * running_correct_gender / running_total_gender
            loss_age = running_loss_age * 32 / running_total_age
            accuracy_age = 100 * running_correct_age / running_total_age

            acc_gender /= count
            acc_age /= count
            micro_gender /= count
            micro_age /= count
            macro_gender /= count
            macro_age /= count

            if acc_gender > best_acc_gender:
                best_acc_gender = acc_gender
            if acc_age > best_acc_age:
                best_acc_age = acc_age
            if micro_gender > best_micro_gender:
                best_micro_gender = micro_gender
            if micro_age > best_micro_age:
                best_micro_age = micro_age
            if macro_gender > best_macro_gender:
                best_macro_gender = macro_gender
            if macro_age > best_macro_age:
                best_macro_age = macro_age
            if accuracy_gender > best_pred_gender:
                best_pred_gender = accuracy_gender
            if accuracy_age > best_pred_age:
                best_pred_age = accuracy_age

            writer.add_scalar('scalar/loss_gender_val', loss_gender, epoch)
            writer.add_scalar('scalar/loss_age_val', loss_age, epoch)
            writer.add_scalar('scalar/accuracy_gender_val', accuracy_gender, epoch)
            writer.add_scalar('scalar/accuracy_age_val', accuracy_age, epoch)
            writer.add_scalar('scalar/acc_gender_val', acc_gender, epoch)
            writer.add_scalar('scalar/acc_age_val', acc_age, epoch)
            writer.add_scalar('scalar/micro_gender_val', micro_gender, epoch)
            writer.add_scalar('scalar/micro_age_val', micro_age, epoch)
            writer.add_scalar('scalar/macro_gender_val', macro_gender, epoch)
            writer.add_scalar('scalar/macro_age_val', macro_age, epoch)

            print('gender valing',
                  '    Epoch[%d /100],loss = %.6f,accuracy=%.4f %%, acc_gender = %.4f, micro_gender = %.4f, macro_gender = %.4f, running_total=%d,running_correct=%d' %
                  (epoch + 1, loss_gender, accuracy_gender, acc_gender, micro_gender, macro_gender, running_total_gender, running_correct_gender))
            print('age    valing',
                  '    Epoch[%d /100],loss = %.6f,accuracy=%.4f %%, acc_age = %.4f, micro_age = %.4f, macro_age = %.4f,running_total=%d,running_correct=%d' %
                  (epoch + 1, loss_age, accuracy_age, acc_age, micro_age, macro_age, running_total_age, running_correct_age))


if __name__ == "__main__":
    main()
    writer.close()
