import os
import pandas as pd
import numpy as np

import pdb
import sys, pickle
import argparse, yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import numpy as np
from analysis import Logger, AverageMeter, accuracy, binary_accuracy

from CUB.dataset import load_data, find_class_imbalance, find_partition_indices_by_IG, find_partition_indices_byRandom
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE
from CUB.models import ModelXtoCY, ModelXtoChat_ChatToY, ModelXtoY, ModelXtoC, ModelOracleCtoY, ModelXtoCtoY
# root_dir = "CUB/output/mi"
# subset_sizes = ["1",'2',"4",'6','8','10']
# report = pd.DataFrame(columns = ["Task (y) Error", "Concept Dimensions", "Residue Dimensions"])
# for dir in os.listdir(root_dir):
#     residue_dir = os.path.join(root_dir, dir)
#     for subset_dir in os.listdir(residue_dir):
#         if subset_dir in subset_sizes:
#             temp = []
#             data_path = os.path.join(residue_dir, subset_dir)
#             data = np.loadtxt(data_path + "/tti_results.txt", delimiter=" ")
#             temp.extend([(100-data[0][1])/100, int(subset_dir), int(dir.strip("_residue"))])
#             report.loc[len(report)] = temp
# report = report.astype({"Concept Dimensions": int, 
#                         "Residue Dimensions": int})
# print(report)
# report.to_csv("CUB/output/experiment_results/" + "mi_results_tuned_100.csv")


# root_dir = "CUB/output/detach_model_subset_concepts"
# subset_sizes = ['1', '2','4', '6','8','10','12','16','20']
# report = pd.DataFrame(columns = ["Pearson Correlation Coefficient", "Task (y) Error", "Concept Dimensions", "Residue Dimensions"])
# for dir in os.listdir(root_dir):
#     residue_dir = os.path.join(root_dir, dir)
#     for subset_dir in os.listdir(residue_dir):
#         if subset_dir in subset_sizes:
#             temp = []
#             data_path = os.path.join(residue_dir, subset_dir)
#             data = np.loadtxt(data_path + "/results.txt", delimiter=" ")
#             temp.extend([data[4], data[0], int(subset_dir), int(dir.strip("_residue"))])
#             report.loc[len(report)] = temp
# report = report.astype({"Concept Dimensions": int, 
#                         "Residue Dimensions": int})
# print(report)
# report.to_csv("CUB/output/experiment_results/" + "detached_results.csv")

# root_dir = "CUB/output/mi"
# subset_sizes = ["1",'2',"4",'6','8','10']
# report = pd.DataFrame(columns = ["Pearson Correlation Coefficient", "Task (y) Error", "Concept Dimensions", "Residue Dimensions"])
# for dir in os.listdir(root_dir):
#     residue_dir = os.path.join(root_dir, dir)
#     for subset_dir in os.listdir(residue_dir):
#         if subset_dir in subset_sizes:
#             temp = []
#             data_path = os.path.join(residue_dir, subset_dir)
#             data = np.loadtxt(data_path + "/results_7_None_50.txt", delimiter=" ")
#             temp.extend([data[4], data[0], int(subset_dir), int(dir.strip("_residue"))])
#             report.loc[len(report)] = temp
# report = report.astype({"Concept Dimensions": int, 
#                         "Residue Dimensions": int})
# print(report)
# report.to_csv("CUB/output/experiment_results/" + "mi_results_tuned_7_None_50.csv")

# root_dir = "CUB/output/mi"
# subset_sizes = ['6']
# report = pd.DataFrame(columns = ["Epoch", "Loss", "Residue-Concept"])
# for dir in os.listdir(root_dir):
#     residue_dir = os.path.join(root_dir, dir)
#     for subset_dir in os.listdir(residue_dir):
#         if subset_dir in subset_sizes:
#             data_path = os.path.join(residue_dir, subset_dir, "7-64_log.txt")
#             epoch = 0
#             with open(data_path) as f:
#                 for line in f:
#                     if line[0] == "E":
#                         list = line.split("\t")
#                         loss_block = list[6]
#                         loss = loss_block.split(" ")[3]
#                         if epoch >= 90:
#                             report.loc[len(report)] = [epoch, loss, f'{dir[0]}-{subset_dir}']   
#                         epoch += 1                     
# print(report)
# report.to_csv("CUB/output/experiment_results/" + "mi_training_loss.csv")

# size = 10
# top_indices = find_partition_indices(size)
# indices = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
#     93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
#     183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
#     254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]

# index = [indices[i] for i in top_indices]

# uncertainties = np.array([0]*112)
# data = pickle.load(open(os.path.join("CUB_processed/class_attr_data_10", 'train.pkl'), 'rb'))
# for d in data:
#     uncertainties += np.array(d['attribute_certainty'])[indices]
# uncertainties = uncertainties/len(data)
# uncertainties_indices = np.argpartition(uncertainties,-size)[-size:]
# print(top_indices, uncertainties_indices)

# residual = 32
# reduce = ''
# report = pd.DataFrame(columns = ["Concept", "Residuals", "Experiments", "Task Error", "Task Error std", "Concept Error", "Concept Error std", "Cross Correlation", "Cross Correlation std", "Mutual Information", "Mutual Information std"])
# rootDir = f"/home/shelton/supervised-concept/CUB/output/manuscript_new/CUB/{residual}r{reduce}"
# concepts = [20, 40, 60, 80, 100, 112]
# for exp in os.listdir(rootDir):
#     if exp == "feature_attributions.csv":
#         continue
#     if exp == "detached_c":
#         continue
#     for concept in concepts:
#         dataDir = os.path.join(rootDir, exp, str(concept), 'results.txt')
#         data = np.loadtxt(dataDir, delimiter=" ")

#         temp = []
#         result = [concept, residual, exp, data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]]
#         temp.extend(result)
#         report.loc[len(report)] = temp
#         report.to_csv(f"/home/shelton/supervised-concept/CUB/output/manuscript_new/CUB/residual{residual}{reduce}_results.csv")
# print(report)

residual = 32
reduce = ''
report = pd.DataFrame(columns = ["Concept", "Residuals", "Experiments", "Task Error", "Task Error std", "Percent of Concepts Replaced"])
rootDir = f"/home/shelton/supervised-concept/CUB/output/manuscript_new/CUB/{residual}r{reduce}"
intervention = "pos"
concepts = [20, 40, 60, 80, 100, 112]
for exp in os.listdir(rootDir):
    if exp == "feature_attributions.csv":
        continue
    if exp == "detached_c":
        continue
    for concept in concepts:
        dataDir = os.path.join(rootDir, exp, str(concept), f'{intervention}TTI.txt')
        data = np.loadtxt(dataDir, delimiter=" ")
        for index in range(len(data)):
            result = [concept, residual, exp, data[index][0], data[index][1], (index+1)/10]
            report.loc[len(report)] = result
report.to_csv(f"/home/shelton/supervised-concept/CUB/output/manuscript_new/CUB/residual{residual}_{reduce}_{intervention}TTI_results.csv")
print(report)