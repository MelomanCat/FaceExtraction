import cv2
import dlib

# Загрузка детектора лиц из dlib
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

# Открытие видеофайла для чтения
cap = cv2.VideoCapture('183.mp4')

# Считывание кадров из видео и детекция лиц
frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Детекция лиц на текущем кадре
    dets = cnn_face_detector(frame, 1)

    # Отрисовка результатов детекции лиц на текущем кадре
    for i, d in enumerate(dets):
        x, y, w, h = d.rect.left(), d.rect.top(), d.rect.width(), d.rect.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Сохранение текущего кадра с результатами детекции лиц в файл
    output_folder = "C:/Users/Olga/Desktop/Python/pythonProject/datasets/original_sequences/youtube/raw/videos"
    print("Saving file to:", output_folder)
    output_filename = output_folder + 'output_frame_{:04d}.jpg'.format(frame_num)
    cv2.imwrite(output_filename, frame)
    frame_num += 1

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()

