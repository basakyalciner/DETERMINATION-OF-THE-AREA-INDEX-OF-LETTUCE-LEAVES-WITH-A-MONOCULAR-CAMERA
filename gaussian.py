import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

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

def cluster_with_gaussian(largest_areas, n_clusters):
    # Verileri numpy dizisine dönüştürüyoruz
    data = np.array(largest_areas).reshape(-1, 1)

    # Gaussian Mixture Model oluşturuyoruz
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)

    # Verileri GMM'ye uyduruyoruz
    gmm.fit(data)

    # GMM ile verileri tahmin ediyoruz
    predicted_labels = gmm.predict(data)

    # Kümeleme sonuçlarını en büyük piksel değerine göre sıralıyoruz
    cluster_order = np.argsort(gmm.means_.flatten())

    # Sıralamayı kullanarak küme etiketlerini güncelliyoruz
    new_labels = np.zeros_like(predicted_labels)
    for i, label in enumerate(cluster_order):
        new_labels[predicted_labels == label] = i

    return new_labels

if __name__ == "__main__":
    folder_path1 = r'C:\Users\bskylcnr\Desktop\test1\Lugano'
    folder_path2 = r'C:\Users\bskylcnr\Desktop\test1\Aphylion'

    # Verileri kümeleme için kullanacağımız liste
    lugano_largest_areas = get_largest_areas(folder_path1)
    aphylion_largest_areas = get_largest_areas(folder_path2)

    # Kümeleme için küme sayısını belirleyelim (örn: 7)
    n_clusters = 7

    # Gaussian Mixture Model ile kümeleme yapalım
    predicted_labels_lugano = cluster_with_gaussian(lugano_largest_areas, n_clusters)
    predicted_labels_aphylion = cluster_with_gaussian(aphylion_largest_areas, n_clusters)

    # Her küme için verileri saklamak üzere boş listeler
    clusters_lugano = [[] for _ in range(n_clusters)]
    clusters_aphylion = [[] for _ in range(n_clusters)]

    # Verileri kümelere göre grupla
    for i, label in enumerate(predicted_labels_lugano):
        clusters_lugano[label].append(lugano_largest_areas[i])
    print(clusters_lugano[::])
    
    for i, label in enumerate(predicted_labels_aphylion):
        clusters_aphylion[label].append(aphylion_largest_areas[i])
    print(clusters_aphylion[::])
    # Kümelerin renkleri için farklı renkler seçelim
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Grafik çizimi
    plt.figure(figsize=(16, 6))

    # Combined Growth Graph
    plt.subplot(1, 1, 1)
    for cluster_idx, cluster_data in enumerate(clusters_lugano):
        plt.scatter(range(1, len(cluster_data) + 1), cluster_data, color=colors[cluster_idx], label=f'Lugano Cluster {cluster_idx + 1}')
    
    for cluster_idx, cluster_data in enumerate(clusters_aphylion):
        plt.scatter(range(1, len(cluster_data) + 1), cluster_data, color=colors[cluster_idx], label=f'Aphylion Cluster {cluster_idx + 1}', marker='*')
    
    plt.ylabel('Pixel Values')
    plt.xlabel('Lettuce Number')  # Yeni eklenen başlık
    plt.title("1-7 Weeks Growth Graph of Lettuces")
    plt.legend()

    plt.tight_layout()
    plt.show()
# Bu kod parçası, her iki çeşidin (Lugano ve Aphylion) büyüme grafiklerini tek bir grafikte birleştirir. marker parametresi kullanılarak Aphylion verileri için farklı bir işaretçi (örneğin 'x') kullanılır.





