import cv2
import dlib

# Path ke file berikut harus disesuaikan dengan lokasi di sistem Anda
shape_predictor_path = "./shape_predictor_68_face_landmarks.dat"

# Fungsi untuk mendeteksi landmark tubuh
def detect_body_landmarks(image_path):
    # Inisialisasi detector wajah dan pose predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)

    # Baca gambar
    image = cv2.imread(image_path)

    # Ubah gambar menjadi grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = detector(gray)

    # Loop melalui setiap wajah yang terdeteksi
    for face in faces:
        # Prediksi pose dan lokasi landmark wajah
        landmarks = predictor(gray, face)

        # Loop melalui setiap landmark
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y

            # Gambar lingkaran di sekitar setiap landmark
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # Tampilkan gambar dengan landmark
    cv2.imshow("Body Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Jalankan fungsi deteksi landmark tubuh
image_path = "./trainingData/2.png"  # Ubah dengan path gambar Anda
detect_body_landmarks(image_path)
