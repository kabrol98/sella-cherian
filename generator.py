from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

x_array = []

db_scores_kmeans = []
ch_scores_kmeans = []
s_scores_kmeans = []
cluster_nums_kmeans = []

db_scores_em = []
ch_scores_em = []
s_scores_em = []
cluster_nums_em = []

db_scores_dbscan = []
ch_scores_dbscan = []
s_scores_dbscan = []
cluster_nums_dbscan = []

db_scores_optics = []
ch_scores_optics = []
s_scores_optics = []
cluster_nums_optics = []

file = open("result.txt", "r")
for line in file:
    if line[-1] == "\n":
        line = line[:-1]

    if "kmeans" in line:
        split_line = line.split(" ")
        db_scores_kmeans.append(float(split_line[1]))
        ch_scores_kmeans.append(float(split_line[2]))
        s_scores_kmeans.append(float(split_line[3]))
        cluster_nums_kmeans.append(int(split_line[4]))
    elif "em" in line:
        split_line = line.split(" ")
        db_scores_em.append(float(split_line[1]))
        ch_scores_em.append(float(split_line[2]))
        s_scores_em.append(float(split_line[3]))
        cluster_nums_em.append(int(split_line[4]))
    elif "dbscan" in line:
        split_line = line.split(" ")
        db_scores_dbscan.append(float(split_line[1]))
        ch_scores_dbscan.append(float(split_line[2]))
        s_scores_dbscan.append(float(split_line[3]))
        cluster_nums_dbscan.append(int(split_line[4]))
    elif "optics" in line:
        split_line = line.split(" ")
        db_scores_optics.append(float(split_line[1]))
        ch_scores_optics.append(float(split_line[2]))
        s_scores_optics.append(float(split_line[3]))
        cluster_nums_optics.append(int(split_line[4]))
    else:
        x_array.append(int(line))

file.close()

f, axes = plt.subplots(4,1)

plt.suptitle("Cluster Method Performance")

ax = axes[0]
ax.plot(x_array, db_scores_kmeans, 'r')
ax.plot(x_array, db_scores_em, 'b')
ax.plot(x_array, db_scores_dbscan, 'g')
ax.plot(x_array, db_scores_optics, 'y')
ax.set_ylabel("davies bouldin", fontsize=6)
ax.set_xticklabels([])
red_patch = mpatches.Patch(color='red', label='kmeans')
ax.legend(handles=[red_patch])

ax = axes[1]
ax.plot(x_array, ch_scores_kmeans, 'r')
ax.plot(x_array, ch_scores_em, 'b')
ax.plot(x_array, ch_scores_dbscan, 'g')
ax.plot(x_array, ch_scores_optics, 'y')
ax.set_ylabel("calinski harabasz", fontsize=6)
ax.set_xticklabels([])
blue_patch = mpatches.Patch(color='blue', label='em')
ax.legend(handles=[blue_patch])

ax = axes[2]
ax.plot(x_array, s_scores_kmeans, 'r')
ax.plot(x_array, s_scores_em, 'b')
ax.plot(x_array, s_scores_dbscan, 'g')
ax.plot(x_array, s_scores_optics, 'y')
ax.set_ylabel("silhouette", fontsize=6)
ax.set_xticklabels([])
green_patch = mpatches.Patch(color='green', label='dbscan')
ax.legend(handles=[green_patch])

ax = axes[3]
ax.plot(x_array, cluster_nums_kmeans, 'r')
ax.plot(x_array, cluster_nums_em, 'b')
ax.plot(x_array, cluster_nums_dbscan, 'g')
ax.plot(x_array, cluster_nums_optics, 'y')
ax.set_ylabel("cluster nums", fontsize=6)
yellow_patch = mpatches.Patch(color='yellow', label='optics')
ax.legend(handles=[yellow_patch])

f.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('compare.png')

# plt.plot()
    # plot_title=f'{N}_columns|{DATA_TYPE}|{SUMMARY_TYPE}|{CLUSTER_TYPE}|{pkey}'
    # path_name=f'{N}_columns-{DATA_TYPE}-{SUMMARY_TYPE}-{CLUSTER_TYPE}-{pkey}'
    # plt.suptitle(plot_title)
    # plt.savefig(f'testing/cluster_results/{path_name}.png')
    # print(f'Saved figure {plot_title} to {path_name}')
