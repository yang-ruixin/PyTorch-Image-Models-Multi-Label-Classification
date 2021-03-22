""" Multi-Label Model

YANG Ruixin
2021/03/18
https://github.com/yang-ruixin
yang_ruixin@126.com (in China)
rxn.yang@gmail.com (out of China)
"""

from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn as nn
import warnings

# ================================
# define weights here
WEIGHT_COLOR = WEIGHT_GENDER = WEIGHT_ARTICLE = 1 / 3
# ================================


class MultiLabelModel(nn.Module):
    def __init__(self, model, n_color_classes, n_gender_classes, n_article_classes):
        super().__init__()
        self.base_model = model.as_sequential_for_ML()
        last_channel = model.num_features
        # print('================================')
        # print('type(self.base_model)', type(self.base_model))
        # print('last_channel', last_channel)
        # print('================================')

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        self.color = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_color_classes)
        )
        self.gender = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_gender_classes)
        )
        self.article = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_article_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        # print('================================')
        # print('x.shape', x.shape)
        # print(self.gender(x).shape)
        # print('================================')

        return {
            'color': self.color(x),
            'gender': self.gender(x),
            'article': self.article(x)
        }

    @staticmethod
    def get_loss(loss_fn, output, target):
        loss_color = loss_fn(output['color'], target['color_labels'].cuda())
        loss_gender = loss_fn(output['gender'], target['gender_labels'].cuda())
        loss_article = loss_fn(output['article'], target['article_labels'].cuda())

        loss = WEIGHT_COLOR * loss_color + WEIGHT_GENDER * loss_gender + WEIGHT_ARTICLE * loss_article
        return loss

    @staticmethod
    def get_accuracy(accuracy, output, target, topk=(1,)):
        acc1_color, acc5_color = accuracy(output['color'], target['color_labels'].cuda(), topk=topk)
        acc1_gender, acc5_gender = accuracy(output['gender'], target['gender_labels'].cuda(), topk=topk)
        acc1_article, acc5_article = accuracy(output['article'], target['article_labels'].cuda(), topk=topk)

        acc1 = WEIGHT_COLOR * acc1_color + WEIGHT_GENDER * acc1_gender + WEIGHT_ARTICLE * acc1_article
        acc5 = WEIGHT_COLOR * acc5_color + WEIGHT_GENDER * acc5_gender + WEIGHT_ARTICLE * acc5_article
        return acc1, acc5, {'color': acc1_color, 'gender': acc1_gender, 'article': acc1_article}

    @staticmethod
    def calculate_metrics(output, target):
        predicted_color = output['color'].cpu().argmax(1)
        gt_color = target['color_labels'].cpu()

        predicted_gender = output['gender'].cpu().argmax(1)
        gt_gender = target['gender_labels'].cpu()
        # print('================================')
        # print('predicted_gender', predicted_gender)
        # print('gt_gender', gt_gender)
        # print('================================')

        predicted_article = output['article'].cpu().argmax(1)
        gt_article = target['article_labels'].cpu()

        with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
            warnings.simplefilter("ignore")
            accuracy_color = balanced_accuracy_score(y_true=gt_color.numpy(), y_pred=predicted_color.numpy())
            accuracy_gender = balanced_accuracy_score(y_true=gt_gender.numpy(), y_pred=predicted_gender.numpy())
            accuracy_article = balanced_accuracy_score(y_true=gt_article.numpy(), y_pred=predicted_article.numpy())

        return accuracy_color, accuracy_gender, accuracy_article
