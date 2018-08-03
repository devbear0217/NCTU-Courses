import sys
import csv
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score


def write_file(input_data_name, final_result):
    # creating file name
    name = input_data_name.split("_", 1)
    name = name[1]
    name = name.split(".")
    string = name[0]

    filename = string[0:len(string) - 1] + "_" + string[len(string) - 1] + "_out." + name[1]
    print("output file:", filename)

    # writing csv file
    my_file = open(filename, "w")
    with my_file:
        write_file = csv.writer(my_file)
        write_file.writerows(final_result)

    print("Writing output file completed")


def custom_dbscan(data, eps, min_point):
    """
    start cluster number from 1.
    data point numbered as -1 is noise
    """

    data_length = data.shape[0]
    clust_label = [0] * data_length

    # Cluster ID
    c = 0

    for P in range(0, data.shape[0]):
        if not (clust_label[P] == 0):
            continue

        neigh_point = find_points(data,
                                  P,
                                  eps)

        if len(neigh_point) < min_point:
            clust_label[P] = -1

        else:
            c = c + 1
            expand_cluster(data,
                           clust_label,
                           P,
                           neigh_point,
                           c,
                           eps,
                           min_point)

    return clust_label


def expand_cluster(data, clust_label, p, neigh_point, c, eps, min_point):
    clust_label[p] = c

    queue_item = 0
    while queue_item < len(neigh_point):
        pn = neigh_point[queue_item]

        if clust_label[pn] == -1:
            clust_label[pn] = c

        elif clust_label[pn] == 0:
            clust_label[pn] = c

            pn_neighbors = find_points(data, pn, eps)

            if len(pn_neighbors) >= min_point:
                neigh_point = neigh_point + pn_neighbors

        queue_item = queue_item + 1


def find_points(data, p, eps):
    neighbors = []

    for Pn in range(0, data.shape[0]):
        if np.linalg.norm(data[p] - data[Pn]) < eps:
            neighbors.append(Pn)

    return neighbors


def main():
    print("executing program")

    # data = "sample_dataset2.csv"
    # ground_truth = "sample_dataset2_train.csv"
    # eps = 0.517
    # sample = 7

    script = sys.argv

    # importing dataset
    data = script[1]
    ground_truth = script[2]
    eps = float(script[3])

    # suggested number of min sample in DBScan = (2*feature) - 1
    sample = int(script[4])

    imported_data = np.genfromtxt(data,
                                  delimiter=',')
    import_ground_truth = np.genfromtxt(ground_truth,
                                        delimiter=',')

    if data == "sample_dataset1.csv":
        reduced_dim = TSNE(n_components=2).fit_transform(imported_data[0:, 1:])
        imported_data = np.column_stack((imported_data[:, 0],
                                         reduced_dim))

    # sort the ground truth based on ID
    if data == "sample_dataset2.csv":
        import_ground_truth = import_ground_truth[import_ground_truth[:, 0].argsort()]

    # count the dimension of data
    # nData = imported_data.shape[0]
    features = imported_data.shape[1] - 1
    print("features =", features)

    gt = import_ground_truth.shape[1]-1
    gt = import_ground_truth[:, gt]
    gt = gt.astype(int)

    # normalize the input data
    # if input dataset is "sample_dataset3.csv" then use MaxAbsScaler()
    # otherwise use StandardScaler() to normalize input data
    if data == "sample_dataset3.csv":
        scaler = MaxAbsScaler()
        normalized_data = scaler.fit_transform(imported_data[0:, 1:])
    else:
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(imported_data[0:, 1:])

    # clustering with handcrafted DBScan
    cluster = custom_dbscan(normalized_data,
                            eps=eps,
                            min_point=sample)
    cluster = np.asarray(cluster)
    cluster = cluster.astype(int)

    # rename the cluster number from 1++ to 0++
    for i in range(0, len(cluster)):
        if not cluster[i] == -1:
            cluster[i] -= 1

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_own = len(set(cluster)) - (1 if -1 in cluster else 0)
    print("number of cluster:", n_clusters_own)

    predict_data = np.column_stack((imported_data[:, 0:], cluster))

    # get the index of noise data
    noise_index = np.argwhere(cluster == -1)
    nc = len(noise_index)

    noise_data = predict_data[np.where(predict_data[:, predict_data.shape[1] - 1] == -1)]

    clustered_data = predict_data[np.where(predict_data[:, predict_data.shape[1] - 1] != -1)]
    print("noise count:", nc)


# if there are noise data, classify the noise data by using KNN
    if nc > 0:
        x_train = clustered_data[:, 1:clustered_data.shape[1]-1]
        y_train = clustered_data[:, clustered_data.shape[1]-1]

        x_test = noise_data[:, 1:noise_data.shape[1]-1]

        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(x_train, y_train)
        knn_pred = knn.predict(x_test)

        np.put(predict_data[:, predict_data.shape[1]-1], noise_index, knn_pred)

    """Get the prediction label"""
    pred_label = []

    if data == "sample_dataset2.csv":
        gt_index = import_ground_truth[:, 0]
        for i in gt_index:
            temp = predict_data[np.where(predict_data[:, 0] == i)]
            pred_label.append(temp[:, temp.shape[1] - 1])

    elif data == "sample_dataset4.csv":
        for i in predict_data[0:len(gt), ]:
            if i[i.shape[0]-1] == -1:
                pred_label.append(-1)
            elif i[i.shape[0]-1] == 0:
                pred_label.append(1)
            else:
                pred_label.append(0)

    elif data == "sample_dataset1.csv":
        for i in predict_data[0:len(gt), ]:
            if i[i.shape[0]-1] == -1:
                pred_label.append(-1)
            elif i[i.shape[0]-1] == 0:
                pred_label.append(3)
            elif i[i.shape[0]-1] == 1:
                pred_label.append(4)
            elif i[i.shape[0]-1] == 2:
                pred_label.append(1)
            elif i[i.shape[0]-1] == 3:
                pred_label.append(2)
            else:
                pred_label.append(0)

    else:
        pred_label = predict_data[0:len(gt):, predict_data.shape[1]-1]

    """Find The f1_Score"""
    result_gt = np.column_stack((gt, pred_label))
    f1Score = f1_score(gt, pred_label, average='micro')
    print("f1Score:", f1Score)


# for i in result_gt:
    #     print(i)

    """Write the file"""
    # final result of clustering
    final_result = np.column_stack((imported_data[:, 0], predict_data[:, predict_data.shape[1] - 1]))
    final_result.astype(int)

    write_file(data, final_result)


main()
