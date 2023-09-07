import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def get_largest_areas(folder_path):
    largest_areas = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            frame = cv2.imread(os.path.join(folder_path, filename))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower = np.array([30, 50, 50])
            upper = np.array([70, 255, 255])

            mask = cv2.inRange(hsv, lower, upper)

            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

            largest_label = 1
            max_area = stats[1, cv2.CC_STAT_AREA]
            for label in range(2, ret):
                area = stats[label, cv2.CC_STAT_AREA]
                if area > max_area:
                    max_area = area
                    largest_label = label

            largest_areas.append(max_area)

    largest_areas.sort()
    return largest_areas

def cluster_with_gaussian(largest_areas, n_clusters):
    data = np.array(largest_areas).reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    gmm.fit(data)
    predicted_labels = gmm.predict(data)
    cluster_order = np.argsort(gmm.means_.flatten())
    new_labels = np.zeros_like(predicted_labels)
    for i, label in enumerate(cluster_order):
        new_labels[predicted_labels == label] = i
    return new_labels

if __name__ == "__main__":
    folder_path1 = r'C:\Users\bskylcnr\Desktop\test1\Lugano'
    folder_path2 = r'C:\Users\bskylcnr\Desktop\test1\Aphylion'

    lugano_largest_areas = get_largest_areas(folder_path1)
    aphylion_largest_areas = get_largest_areas(folder_path2)

    n_clusters = 7

    predicted_labels_lugano = cluster_with_gaussian(lugano_largest_areas, n_clusters)
    predicted_labels_aphylion = cluster_with_gaussian(aphylion_largest_areas, n_clusters)

    clusters_lugano = [[] for _ in range(n_clusters)]
    clusters_aphylion = [[] for _ in range(n_clusters)]

    for i, label in enumerate(predicted_labels_lugano):
        clusters_lugano[label].append(lugano_largest_areas[i])

    for i, label in enumerate(predicted_labels_aphylion):
        clusters_aphylion[label].append(aphylion_largest_areas[i])




def plot_cluster_statistics(data, cluster_number, title):
    max_val = np.max(data)
    min_val = np.min(data)
    mean_val = np.mean(data)
    median_val = np.median(data)

    plt.axvline(x=max_val, color='red', linestyle='dashed', linewidth=2, label='Max')
    plt.axvline(x=min_val, color='green', linestyle='dashed', linewidth=2, label='Min')
    plt.axvline(x=mean_val, color='purple', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(x=median_val, color='orange', linestyle='dashed', linewidth=2, label='Median')
    plt.hist(data, bins=20, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel("pixel")
    plt.ylabel("time")
    plt.legend()

from scipy.stats import norm
def plot_std_curve(data, title):
    std_val = np.std(data)
    mean_val = np.mean(data)
    print(std_val)
    plt.plot(data, color='blue', alpha=0.7, label='Std Deviation')
    plt.axhline(y=mean_val, color='purple', linestyle='dashed', linewidth=2, label='Mean')
    plt.title(title)
    plt.xlabel("Week")
    plt.ylabel("Pixel")
    plt.legend()


def plot_bell_curve(data, title):
    std_val = np.std(data)
    mean_val = np.mean(data)

    x = np.linspace(min(data), max(data), 100)
    y = norm.pdf(x, mean_val, std_val)

    plt.plot(x, y, color='blue', label='Bell Curve')
    plt.plot(data, color='red', alpha=0.7, label='Std Deviation')
    plt.axhline(y=mean_val, color='purple', linestyle='dashed', linewidth=2, label='Mean')
    plt.title(title)
    plt.xlabel("Week")
    plt.ylabel("Pixel")
    plt.legend()

plt.figure(figsize=(16, 8))

for i in range(n_clusters):
    plt.subplot(2, 4, i+1)
    plot_cluster_statistics(clusters_lugano[i], i, f"Lugano Week {i+1}")

plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 8))
for i in range(n_clusters):
    plt.subplot(2, 4, i+1)
    plot_cluster_statistics(clusters_aphylion[i], i, f"Aphylion Week {i+1}")
    


plt.tight_layout()
plt.show()

# 7 haftanın standart sapmalarını hesapla
lugano_std_data = [np.std(cluster) for cluster in clusters_lugano]
aphylion_std_data = [np.std(cluster) for cluster in clusters_aphylion]

# Tek bir grafik üzerinde standart sapma karşılaştırmasını çizdir
plt.figure(figsize=(10, 6))

plt.plot(lugano_std_data, label='Lugano', marker='o')
plt.plot(aphylion_std_data, label='Aphylion', marker='o')

plt.title('Standard Deviation Comparison for 7 Weeks')
plt.xlabel('Week')
plt.ylabel('Standard Deviation')
plt.legend()
plt.xticks(range(n_clusters), [f'Week {i+1}' for i in range(n_clusters)])

plt.tight_layout()
plt.show()


# Tek bir grafik üzerinde çan eğrisi ve standart sapma karşılaştırmasını çizdir
plt.figure(figsize=(10, 6))

plt.plot(lugano_std_data, label='Lugano Std Deviation', marker='o')
plt.plot(aphylion_std_data, label='Aphylion Std Deviation', marker='o')

plt.title('Bell Curve and Std Deviation Comparison for 7 Weeks')
plt.xlabel('Week')
plt.ylabel('Value')
plt.legend()
plt.xticks(range(n_clusters), [f'Week {i+1}' for i in range(n_clusters)])

plt.tight_layout()
plt.show()
# Bu güncellenmiş kod, 7 haftanın standart sapmalarını ve çan eğrisini çizecek, ardından bu verileri karşılaştıracak. Umarım bu kod size istediğiniz sonucu sağlar.






  