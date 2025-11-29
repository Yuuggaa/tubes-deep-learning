# load datset from directory
def load_dataset_from_directory(directory):
    import os
    import cv2
    import numpy as np

    images = []
    labels = []
    label_names = os.listdir(directory)
    label_map = {name: idx for idx, name in enumerate(label_names)}

    for label_name in label_names:
        label_dir = os.path.join(directory, label_name)
        if not os.path.isdir(label_dir):
            continue
        for filename in os.listdir(label_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(label_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(label_map[label_name])

    return np.array(images), np.array(labels), label_map