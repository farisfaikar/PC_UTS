import cv2
import numpy as np
from matplotlib import pyplot as plt


def rotate_image():
    img = cv2.imread('image.jpg')
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    cv2.imshow("Rotated Image", rotated)
    cv2.waitKey(0)


def resize_image():
    img = cv2.imread('image.jpg')
    resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Resized Image", resized)
    cv2.waitKey(0)


def crop_image():
    img = cv2.imread('image.jpg')
    cropped = img[150:150+300, 100:100+200]
    cv2.imshow("Cropped Image", cropped)
    cv2.waitKey(0)


def flip_image():
    img = cv2.imread('image.jpg')
    flipped = cv2.flip(img, 1)
    cv2.imshow("Flipped Image", flipped)
    cv2.waitKey(0)


def translate_image():
    img = cv2.imread('image.jpg')
    (h, w) = img.shape[:2]
    M = np.float32([[1, 0, 50], [0, 1, 25]])
    translated = cv2.warpAffine(img, M, (w, h))
    cv2.imshow("Translated Image", translated)
    cv2.waitKey(0)


def gaussian():
   # Load the image
    img = cv2.imread('image.jpg')

    # Apply Gaussian blur with kernel size of 5x5 and sigma value of 0
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Display the blurred image
    cv2.imshow('Gaussian Blurred Image', blur)

    # Wait for a key press and then close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def operasi_piksel_dan_histogram():
    # membaca gambar
    img = cv2.imread('image.jpg')

    # konversi dari BGR ke RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # menambahkan kontras dan kecerahan
    alpha = 1.5
    beta = 50
    img_corrected = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # menampilkan histogram gambar
    plt.hist(img_corrected.ravel(), 256, [0, 256])
    plt.show()

    # menampilkan gambar asli dan hasil koreksi warna
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(img_corrected)
    plt.title('Corrected Image')

    plt.show()


def operasi_ketetanggaan_piksel():
    # Membaca gambar
    img = cv2.imread('image.jpg')

    # Konversi ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Operasi ketetanggaan piksel
    blur = cv2.medianBlur(gray, 5)

    # Menampilkan gambar hasil
    cv2.imshow('Blurred Image', blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def operasi_morfologi(iterasi):
    # Baca gambar
    img = cv2.imread('image.jpg')

    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Lakukan operasi thresholding pada gambar
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Buat kernel untuk operasi morfologi
    kernel = np.ones((5,5), np.uint8)

    # Lakukan operasi erosi
    erosion = cv2.erode(thresh, kernel, iterations=iterasi)

    # Lakukan operasi dilasi
    dilation = cv2.dilate(thresh, kernel, iterations=iterasi)

    # Tampilkan hasil operasi morfologi
    cv2.imshow(f'Erosion iterasi: {iterasi}', erosion)
    cv2.imshow(f'Dilation iterasi: {iterasi}', dilation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # === Operasi Geometrik ===
    rotate_image()
    # resize_image()
    # crop_image()
    # flip_image()
    # translate_image()

    # === Operasi Kawasan Frekuensi ===
    # gaussian()

    # === Operasi Piksel dan Histogram ===
    # operasi_piksel_dan_histogram()

    # === Operasi Ketetanggaan Piksel ===
    # operasi_ketetanggaan_piksel()
    
    # === Operasi Morfologi ===
    # operasi_morfologi(1)
    # operasi_morfologi(5)


if __name__ == '__main__':
    main()
