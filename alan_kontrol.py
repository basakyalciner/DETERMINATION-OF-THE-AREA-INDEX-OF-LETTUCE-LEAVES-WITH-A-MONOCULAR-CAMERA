import cv2
import numpy as np

frame = cv2.imread(r"C:\Users\bskylcnr\Desktop\test1\Aphylion\5.jpg")
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lower = np.array([30, 50, 50])
upper = np.array([70, 255, 255])

mask = cv2.inRange(hsv, lower, upper)
res = cv2.bitwise_and(frame, frame, mask=mask)

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

# En büyük alanlı nesneyi seçin ve sadece onu göstermek için maskeyi güncelleyin
mask = np.zeros_like(labels, dtype=np.uint8)
mask[labels == largest_label] = 255
# Piksel alanını yazdırın
print("Lettuce pixel area:", max_area)
# Sonucu gösterin
cv2.imshow('frame', frame)
cv2.imshow('res', res)
cv2.imshow('largest_object', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import os
# import glob

# def process_image(image_path):
#     frame = cv2.imread(image_path)
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     lower = np.array([30, 50, 50])
#     upper = np.array([70, 255, 255])

#     mask = cv2.inRange(hsv, lower, upper)
#     res = cv2.bitwise_and(frame, frame, mask=mask)

#     ret, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

#     largest_label = 1
#     max_area = stats[1, cv2.CC_STAT_AREA]
#     for label in range(2, ret):
#         area = stats[label, cv2.CC_STAT_AREA]
#         if area > max_area:
#             max_area = area
#             largest_label = label

#     mask = np.zeros_like(labels, dtype=np.uint8)
#     mask[labels == largest_label] = 255

#     return max_area

# # Define the directory path containing the images
# image_directory = r"C:\Users\bskylcnr\Desktop\test1\Lugano"

# # Get a list of all image files in the directory
# image_files = glob.glob(os.path.join(image_directory, "*.jpg"))

# # Process each image and store the "max_area" values
# max_area_list = []
# for image_file in image_files:
#     max_area = process_image(image_file)
#     max_area_list.append(max_area)

# # Sort the "max_area" values in ascending order
# max_area_list.sort()

# # Print the sorted "max_area" values
# print("Sorted max_area values:")
# for max_area in max_area_list:
#     print(max_area)
