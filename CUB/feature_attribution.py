"""
Evaluate trained models on the official CUB test set
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
import torch
import joblib
import argparse
import numpy as np
from sklearn.metrics import f1_score
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from CUB.dataset import load_data
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES
from analysis import AverageMeter, multiclass_metric, accuracy, binary_accuracy
from torchsummary import summary
from captum.attr import IntegratedGradients
# from captum.attr import LayerConductance
# from captum.attr import NeuronConductance

K = [1, 3, 5] #top k class accuracies to compute

def run_feature_attribution(args):
    model = torch.load(args.model_dirs[0])
    model.eval()

    mlp = model.sec_model
    ig = IntegratedGradients(mlp)

    data_dir = os.path.join(BASE_DIR, args.data_dir, args.eval_data + '.pkl')
    loader = load_data([data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
                       n_class_attr=args.n_class_attr)

    all_outputs, all_targets = [], []
    all_attr_labels, all_attr_outputs, all_attr_outputs_sigmoid, all_attr_outputs2 = [], [], [], []
    all_class_labels, all_class_outputs, all_class_logits = [], [], []
    topk_class_labels, topk_class_outputs = [], []

    feature_attributions = []
    for data_idx, data in enumerate(loader):
        
        inputs, labels, attr_labels = data
        attr_labels = torch.stack(attr_labels).t()  # N x 312

        inputs_var = torch.autograd.Variable(inputs).cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels).cuda() if torch.cuda.is_available() else inputs_var
        
        outputs = model(inputs_var)
        attr_outputs = outputs[1:args.n_attributes+1]
        attr_outputs = torch.cat(attr_outputs, dim=1)

        class_outputs = outputs[0]
        prediction_score, pred_label_idx = torch.topk(class_outputs, 1)
        pred_label_idx = pred_label_idx.squeeze().item()

        attr_outputs.requires_grad_()
        attr, delta = ig.attribute(attr_outputs, target = pred_label_idx, return_convergence_delta=True)
        attr = attr.detach()
        feature_attributions.append(attr)

    return torch.mean(torch.stack(feature_attributions), dim=0).data.cpu().numpy()

if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-log_dir', default='.', help='where results are stored')
    parser.add_argument('-model_dirs', default=None, nargs='+', help='where the trained models are saved')
    parser.add_argument('-model_dirs2', default=None, nargs='+', help='where another trained model are saved (for bottleneck only)')
    parser.add_argument('-eval_data', default='test', help='Type of data (train/ val/ test) to be used')
    parser.add_argument('-use_attr', help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)', action='store_true')
    parser.add_argument('-no_img', help='if included, only use attributes (and not raw imgs) for class prediction', action='store_true')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-n_class_attr', type=int, default=2, help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', default='', help='directory to the data used for evaluation')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES, help='whether to apply bottlenecks to only a few attributes')    
    parser.add_argument('-attribute_group', default=None, help='file listing the (trained) model directory for each attribute group')
    parser.add_argument('-feature_group_results', help='whether to print out performance of individual atttributes', action='store_true')
    parser.add_argument('-use_relu', help='Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    args = parser.parse_args()
    args.batch_size = 1

    print(args)
    attributions = run_feature_attribution(args)
    attributions.tofile(os.path.join(args.log_dir, 'feature_attributions.csv'), sep = ',')
        