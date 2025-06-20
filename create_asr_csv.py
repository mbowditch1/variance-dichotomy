import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
import glob
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Facenet")
parser.add_argument("--backdoor_type", type=str, default="sc")
parser.add_argument("--reinflate", action="store_true")
parser.add_argument("--normalise", action="store_true")
parser.add_argument("--pca_normalise", action="store_true")

args = parser.parse_args()

if args.reinflate:
    run_dir = f"runs/{args.model_name}/reinflate/{args.backdoor_type}/"
    title = f"{args.model_name} {args.backdoor_type} Reinflation ASR"
    fig_name = f"figures/{args.model_name}_{args.backdoor_type}_reinflate_asr.png"
    csv_name = f"csv/{args.model_name}_{args.backdoor_type}_reinflate_asr.csv"
elif args.normalise:
    run_dir = f"runs/{args.model_name}/normalise/{args.backdoor_type}/"
    title = f"{args.model_name} {args.backdoor_type} Normalised ASR"
    fig_name = f"figures/{args.model_name}_{args.backdoor_type}_normalised_asr.png"
    csv_name = f"csv/{args.model_name}_{args.backdoor_type}_normalised_asr.csv"
elif args.pca_normalise:
    run_dir = f"runs/{args.model_name}/pca_normalise/{args.backdoor_type}/"
    title = f"{args.model_name} {args.backdoor_type} PCA Normalised Accuracy"
    fig_name = f"figures/{args.model_name}_{args.backdoor_type}_pca_normalised_asr.png"
    scv_name = f"csv/{args.model_name}_{args.backdoor_type}_pca_normalised_asr.csv"
else:
    run_dir = f"runs/{args.model_name}/standard/{args.backdoor_type}/"
    title = f"{args.model_name} {args.backdoor_type} Standard ASR"
    fig_name = f"figures/{args.model_name}_{args.backdoor_type}_standard_asr.png"
    csv_name = f"csv/{args.model_name}_{args.backdoor_type}_standard_asr.csv"

print(run_dir)
# Go into each run directory, find the accuracy file for each random seed folder
backdoor_log_files = glob.glob(run_dir + "*/backdoor_logs.pkl")

backdoor_logs = []
for log_file in backdoor_log_files:
    with open(log_file, "rb") as f:
        backdoor_logs.append(pickle.load(f))

# Each element is a list of dictoinaries with each dict containing entries for "backdoor_type", and "accuracies"

# For each backdoor we want to average the accuracies of every elemnt of the list apart from the first one, then if its mc we want to subtract this from one

# First subtract from 1 for mc
for backdoor_log in backdoor_logs:
    for log in backdoor_log:
        if log["backdoor_type"] == "sc":
            log["accuracies"] = 1 - np.array(log["accuracies"])

# Now average the accuracies for each backdoor
n_attacks = len(backdoor_logs[0])
mean_accuracies = [np.zeros(n_attacks-2) for _ in range(len(backdoor_logs))]

for m_acc, bl in zip(mean_accuracies, backdoor_logs):
    for i in range(n_attacks-1, 1, -1):
        for j in range(i):
            m_acc[i-2] += bl[j]["accuracies"][-1-(n_attacks-1-i)]

        m_acc[i-2] /= i

overall_mean_accuracies = np.mean(mean_accuracies, axis=0)
overall_max_accuracies = np.max(mean_accuracies, axis=0)
overall_min_accuracies = np.min(mean_accuracies, axis=0)
overall_std_accuracies = np.std(mean_accuracies, axis=0)
overall_var_accuracies = np.var(mean_accuracies, axis=0)

# # Band between max and min
plt.plot(overall_mean_accuracies, label="Mean")
plt.title(title)
plt.xlabel("# Backdoors")
plt.ylabel("Accuracy")
plt.fill_between(range(len(overall_mean_accuracies)), overall_min_accuracies, overall_max_accuracies, alpha=0.3)
plt.savefig(fig_name)
#
# # Write to dat file
with open(csv_name, 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["backdoor", "mean", "max", "min", "std", "var"]
    
    writer.writerow(field)
    for i in range(n_attacks-2):
        writer.writerow([i+1, overall_mean_accuracies[i], overall_max_accuracies[i], overall_min_accuracies[i], overall_std_accuracies[i], overall_var_accuracies[i]])
#
#
# # # Write to dat file
# # with open('data/asr_' + model + '_' + attack_type + '_' + backdoor_type + '.csv', 'w', newline='') as file:
# #     writer = csv.writer(file)
# #     field = ["backdoor", "avg", "min", "max"]
# #     
# #     writer.writerow(field)
# #     for i in range(len(asr)):
# #         writer.writerow([n_attacks[i], asr[i][0], asr[i][1], asr[i][2]])
