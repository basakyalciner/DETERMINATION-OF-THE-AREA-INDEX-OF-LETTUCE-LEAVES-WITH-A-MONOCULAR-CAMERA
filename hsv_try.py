import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
import numpy as np

def get_largest_areas(folder_path):
    largest_areas = []  # Boş bir liste oluşturuyoruz.

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Resmi yükleyin
            frame = cv2.imread(os.path.join(folder_path, filename))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Renk aralığını belirleyin
            lower = np.array([30, 50, 50])
            upper = np.array([70, 255, 255])

            # Maskeyi hesaplayın
            mask = cv2.inRange(hsv, lower, upper)

            # Bağlantı analizi yaparak nesnenin piksel alanını hesaplayın
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

            # En büyük alanı bulmak için istatistikleri kullanın
            largest_label = 1
            max_area = stats[1, cv2.CC_STAT_AREA]
            for label in range(2, ret):
                area = stats[label, cv2.CC_STAT_AREA]
                if area > max_area:
                    max_area = area
                    largest_label = label

            largest_areas.append(max_area)  # Her resmin en büyük alanını listeye ekliyoruz.

    largest_areas.sort()  # Küçükten büyüğe sıralıyoruz
    return largest_areas

if __name__ == "__main__":
    folder_path1 = r'C:\Users\bskylcnr\Desktop\test1\Lugano'
    folder_path2 = r'C:\Users\bskylcnr\Desktop\test1\Aphylion'


# Verileri kümeleme için k-means algoritmasını kullanacağız
kmeans_lugano = KMeans(n_clusters=7, random_state=42)
kmeans_aphylion = KMeans(n_clusters=7, random_state=42)

# Verileri kümelemek için kullanacağımız liste
lugano_largest_areas = get_largest_areas(folder_path1)
aphylion_largest_areas = get_largest_areas(folder_path2)

# Verileri k-means algoritmasıyla kümele
kmeans_lugano.fit([[x] for x in lugano_largest_areas])
kmeans_aphylion.fit([[x] for x in aphylion_largest_areas])

# Kümeleme sonuçları
labels_lugano = kmeans_lugano.labels_
labels_aphylion = kmeans_aphylion.labels_

# Her küme için verileri saklamak üzere boş listeler
clusters_lugano = [[] for _ in range(7)]
clusters_aphylion = [[] for _ in range(7)]

# Verileri kümelere göre grupla
for i, label in enumerate(labels_lugano):
    clusters_lugano[label].append(lugano_largest_areas[i])

for i, label in enumerate(labels_aphylion):
    clusters_aphylion[label].append(aphylion_largest_areas[i])

# Kümelerin renkleri için farklı renkler seçelim
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Grafik çizimi
plt.figure(figsize=(16, 6))

# Lugano Lettuce Growth Graph
plt.subplot(1, 2, 1)
for cluster_idx, cluster_data in enumerate(clusters_lugano):
    plt.scatter(range(1, len(cluster_data) + 1), cluster_data, color=colors[cluster_idx], label=f'Cluster {cluster_idx + 1}')
plt.xlabel('Weeks')
plt.ylabel('Pixel Values')
plt.title("7 Weeks Growth Graph of Lugano Lettuce")
plt.legend()

# Aphylion Growth Graph
plt.subplot(1, 2, 2)
for cluster_idx, cluster_data in enumerate(clusters_aphylion):
    plt.scatter(range(1, len(cluster_data) + 1), cluster_data, color=colors[cluster_idx], label=f'Cluster {cluster_idx + 1}')
plt.xlabel('Weeks')
plt.ylabel('Pixel Values')
plt.title("7 Weeks Growth Graph of Aphylion Lettuce")
plt.legend()

plt.tight_layout()
plt.show()








        
        
        
