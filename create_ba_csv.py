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
    title = f"{args.model_name} {args.backdoor_type} Reinflation Accuracy"
    fig_name = f"figures/{args.model_name}_{args.backdoor_type}_reinflate_accuracy.png"
    csv_name = f"csv/{args.model_name}_{args.backdoor_type}_reinflate_accuracy.csv"
elif args.normalise:
    run_dir = f"runs/{args.model_name}/normalise/{args.backdoor_type}/"
    title = f"{args.model_name} {args.backdoor_type} Normalised Accuracy"
    fig_name = f"figures/{args.model_name}_{args.backdoor_type}_normalised_accuracy.png"
    csv_name = f"csv/{args.model_name}_{args.backdoor_type}_normalised_accuracy.csv"
elif args.pca_normalise:
    run_dir = f"runs/{args.model_name}/pca_normalise/{args.backdoor_type}/"
    title = f"{args.model_name} {args.backdoor_type} PCA Normalised Accuracy"
    fig_name = f"figures/{args.model_name}_{args.backdoor_type}_pca_normalised_accuracy.png"
    scv_name = f"csv/{args.model_name}_{args.backdoor_type}_pca_normalised_accuracy.csv"
else:
    run_dir = f"runs/{args.model_name}/standard/{args.backdoor_type}/"
    title = f"{args.model_name} {args.backdoor_type} Standard Accuracy"
    fig_name = f"figures/{args.model_name}_{args.backdoor_type}_standard_accuracy.png"
    csv_name = f"csv/{args.model_name}_{args.backdoor_type}_standard_accuracy.csv"

print(run_dir)
# Go into each run directory, find the accuracy file for each random seed folder
accuracy_files = glob.glob(run_dir + "*/accuracies.pkl")
backdoor_log_files = glob.glob(run_dir + "*/backdoor_logs.pkl")
print(accuracy_files)

accuracies = []
backdoor_logs = []
for acc_file in accuracy_files:
    with open(acc_file, "rb") as f:
        accuracies.append(pickle.load(f))

for log_file in backdoor_log_files:
    with open(log_file, "rb") as f:
        backdoor_logs.append(pickle.load(f))

# Take the mean of the accuracies
accuracies = np.array(accuracies)
mean_accuracies_split = np.mean(accuracies, axis=0)
max_accuracies_split = np.max(accuracies, axis=0)
min_accuracies_split = np.min(accuracies, axis=0)
var_accuracies_split = np.var(accuracies, axis=0)
std_accuracies_split = np.std(accuracies, axis=0)

mean_accuracies = np.mean(mean_accuracies_split, axis=1)
max_accuracies = np.mean(max_accuracies_split, axis=1)
min_accuracies = np.mean(min_accuracies_split, axis=1)
var_accuracies = np.mean(var_accuracies_split, axis=1)
std_accuracies = np.mean(std_accuracies_split, axis=1)

n_attacks = np.arange(0, len(mean_accuracies))

# Band between max and min
plt.plot(mean_accuracies, label="Mean")
plt.title(title)
plt.xlabel("# Backdoors")
plt.ylabel("Accuracy")
plt.fill_between(range(len(mean_accuracies)), min_accuracies, max_accuracies, alpha=0.3)
plt.savefig(fig_name)

# Write to dat file
with open(csv_name, 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["backdoor", "mean", "max", "min", "variance", "std_dev"]
    
    writer.writerow(field)
    for i in range(len(n_attacks)):
        writer.writerow([n_attacks[i], mean_accuracies[i], max_accuracies[i], min_accuracies[i], var_accuracies[i], std_accuracies[i]])


# # Write to dat file
# with open('data/asr_' + model + '_' + attack_type + '_' + backdoor_type + '.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     field = ["backdoor", "avg", "min", "max"]
#     
#     writer.writerow(field)
#     for i in range(len(asr)):
#         writer.writerow([n_attacks[i], asr[i][0], asr[i][1], asr[i][2]])
