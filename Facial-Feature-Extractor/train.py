import cv2
import numpy as np
from ffextractor import FeaturesLocator
from lazyarray import LazyArray

if __name__ == "__main__":
    locator = FeaturesLocator()
    data_size = 1491
    lazy_images = LazyArray(
        lambda i: cv2.imread(f'./face-training-set/ ({i+1}).jpg'), data_size
    )
    locator.train(lazy_images)

    # Save ASEF filters
    print("\nSaving filters...")
    np.save('results/left_eye_asef', locator.left_eye_asef)
    np.save('results/right_eye_asef', locator.right_eye_asef)
    np.save('results/mouth_asef', locator.mouth_asef)
    np.save('results/nose_asef', locator.nose_asef)
    print("filters saved!")
