import subprocess
import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

cmd = [
    "ffmpeg", "-f", "v4l2",
    "-input_format", "mjpeg",
    "-video_size", "1920x1080",
    "-framerate", "30",
    "-i", "/dev/video0",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-"
]

pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

while True:
    raw = pipe.stdout.read(1920 * 1080 * 3)
    if not raw:
        break

    frame = np.frombuffer(raw, dtype=np.uint8).reshape((1080, 1920, 3)).copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        marker_centers = []
        for idx, marker_id in enumerate(ids.flatten()):
            if marker_id in [0, 1]:
                center = np.mean(corners[idx][0], axis=0).astype(int)
                marker_centers.append(center)
                cv2.circle(frame, tuple(center), 5, (0, 255, 0), -1)

        if len(marker_centers) == 2:
            distance = np.linalg.norm(marker_centers[0] - marker_centers[1])
            print(f"Pixel distance: {distance:.1f}")
            cv2.line(frame, tuple(marker_centers[0]), tuple(marker_centers[1]), (255, 0, 0), 2)

    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipe.terminate()
cv2.destroyAllWindows()