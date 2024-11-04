import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
from scipy.spatial.distance import euclidean # calculate the euclidean distance between two points

def get_mediapipe_app(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    """Initialize and return Mediapipe FaceMesh Solution Graph object"""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return face_mesh


def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio for one eye.

    Args:
        landmarks: (list) Detected landmarks list
        refer_idxs: (list) Index positions of the chosen landmarks
                            in order P1, P2, P3, P4, P5, P6
        frame_width: (int) Width of captured frame
        frame_height: (int) Height of captured frame

    Returns:
        ear: (float) Eye aspect ratio
    """
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points


def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    # Calculate Eye aspect ratio

    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)


def plot_eye_landmarks(frame, left_lm_coordinates, right_lm_coordinates, color):
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame, coord, 2, color, -1)

    # frame = cv2.flip(frame, 1) # 对图像进行翻转
    return frame

def calculate_mouth_ratio(landmarks, refer_idxs, frame_w, frame_h):
    # Calculate Mouth aspecti ratio
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_w, frame_h)
            coords_points.append(coord)
        
        points = np.array(
            [
                [r.x, r.y, r.z, i] for i, r in enumerate(landmarks)
            ]
        )
        month_ratio = euclidean(points[13,:3], points[14,:3]) / euclidean(points[78,:3],points[324,:3])

    except:
        month_ratio = 0.0
        coords_points = None
    # 将张嘴参数和嘴巴坐标返回
    return month_ratio, coords_points

def plot_month_landmarks(frame, coordinates, color):
    if coordinates:
        for coord in coordinates:
            cv2.circle(frame, coord, 2, color, -1)
    
    frame = cv2.flip(frame,1)
    return frame
    
def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image


class VideoFrameHandler:
    def __init__(self):
        """
        Initialize the necessary constants, mediapipe app
        and tracker variables
        """
        # Left and right eye chosen landmarks.
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }

        # mouth chosen landmarks
        self.mouth_idxs = [13,14,78,324]

        # Used for coloring landmark points.
        # Its value depends on the current EAR value.
        self.RED = (0, 0, 255)  # BGR 用于指示警告状态或者疲倦状态
        self.GREEN = (0, 255, 0)  # BGR 用于指示正常状态

        # Initializing Mediapipe FaceMesh solution pipeline
        self.facemesh_model = get_mediapipe_app() # 设置模型并准备进行面部特征点的检测

        # For tracking counters and sharing states in and out of callbacks.
        # 状态追踪，维护与疲劳检测相关的状态信息
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
            "COLOR": self.GREEN,
            "play_eye_alarm": False,

            "yawn_start_time": time.perf_counter(),
            "YAWN_TIME": 0.0, # Holds the amount of time passed with yawn_on(mouth_ratio > 0.2)
            "yawn_on":False,
            "M_COLOR": self.GREEN,
            "play_mouth_alarm": False, # 是否播放警告
        }

        self.EAR_txt_pos = (10, 30) # EAR指标的位置
        self.YAWN_on_txt_pos = (10, 60) # yawn_on指标的位置

    def process(self, frame: np.array, thresholds: dict):
        """
        This function is used to implement our Drowsy detection algorithm

        Args:
            frame: (np.array) Input frame matrix.
            thresholds: (dict) Contains the two threshold values
                               WAIT_TIME and EAR_THRESH.

        Returns:
            The processed frame and a boolean flag to
            indicate if the alarm should be played or not.
        """

        # To improve performance,
        # mark the frame as not writeable to pass by reference.
        # frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape

        offset = 20 # 偏移量
        DROWSY_TIME_txt_pos = (10, int(frame_h // 2 * 1.7)) # 疲倦文本的位置
        ALM_txt_pos = (10, int(frame_h // 2 * 1.85))# 警报文本的位置 
        WARN_txt_pos = (10, ALM_txt_pos[1] + offset)
        results = self.facemesh_model.process(frame) # 使用mp进行处理

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, coordinates = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], frame_w, frame_h)
            mouth_ratio, coords_points = calculate_mouth_ratio(landmarks,self.mouth_idxs, frame_w, frame_h)
            
            # 在帧上绘制眼睛的关键点，颜色基于当前状态
            frame = plot_eye_landmarks(frame, coordinates[0], coordinates[1], self.state_tracker["COLOR"])
            # 在帧上绘制嘴巴的关键点，颜色基于当前的状态
            frame = plot_month_landmarks(frame, coords_points, self.state_tracker["M_COLOR"])
            
            # 对眼睛疲劳状态，疲劳时间的追踪检测
            if EAR < thresholds["EAR_THRESH"]:
                # 累计疲劳的时间
                # Increase DROWSY_TIME to track the time period with EAR less than the threshold
                # and reset the start_time for the next iteration.
                end_time = time.perf_counter() # 记录当前的时间
                self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                # 若累积疲劳时间超过设置的等待时间，则设置播放警报，并在帧上绘制警报文本
                if self.state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_eye_alarm"] = True
                    plot_text(frame, "WAKE UP! WAKE UP", ALM_txt_pos, self.state_tracker["COLOR"])
            # 正常状态，重置时间
            else:
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN
                self.state_tracker["play_eye_alarm"] = False
            # 绘制EAR和DROWSY文本信息
            EAR_txt = f"EAR: {round(EAR, 2)}"
            DROWSY_TIME_txt = f"DROWSY: {round(self.state_tracker['DROWSY_TIME'], 3)} Secs"
            plot_text(frame, EAR_txt, self.EAR_txt_pos, self.state_tracker["COLOR"])
            plot_text(frame, DROWSY_TIME_txt, DROWSY_TIME_txt_pos, self.state_tracker["COLOR"])

            # # 对打哈欠动作，以及持续时长的追踪检测
            # if mouth_ratio > thresholds["MOU_THRESH"]:
            #     if not self.state_tracker["yawn_on"]:
            #         self.state_tracker["yawn_on"] = True # 设置为打哈欠状态
            #         self.state_tracker["yawn_start_time"] = time.perf_counter() # 记录开始打哈欠的开始时间
            #         self.state_tracker["M_COLOR"] = self.RED
            # # 疑似打哈欠结束，判断刚才是否是打哈欠
            # else:
            #     if self.state_tracker["yawn_on"]:
            #         self.state_tracker["YAWN_TIME"] = time.perf_counter() - self.state_tracker["yawn_start_time"]
            #         if 4.500 > self.state_tracker["YAWN_TIME"] > 1.500:
            #             self.state_tracker["yawn_on"] = False
            #             if self.state_tracker["YAWN_TIME"] > 2.:
            #                 self.state_tracker["play_alarm"] = True
            #                 plot_text(frame, "Are you tried!?", ALM_txt_pos, self.state_tracker["COLOR"])
            #     # 正常情况
            #     else:
            #         self.state_tracker["yawn_start_time"] = time.perf_counter()
            #         self.state_tracker["YAWN_TIME"] = 0.0
            #         self.state_tracker["M_COLOR"] = self.GREEN
            #         self.state_tracker["play_alarm"] = False

            # 对打哈欠动作，以及持续时长的追踪检测
            if mouth_ratio > thresholds["MOU_THRESH"]:
                yawn_end_time = time.perf_counter()
                self.state_tracker["YAWN_TIME"] += yawn_end_time - self.state_tracker["yawn_start_time"]
                self.state_tracker["yawn_start_time"] = yawn_end_time
                self.state_tracker["M_COLOR"] = self.RED

                # 打哈欠时间超过2秒，播放warn,并绘制文本
                if self.state_tracker["YAWN_TIME"] > 2.:
                    self.state_tracker["play_mouth_alarm"] = True
                    plot_text(frame, "Are you tried!?", ALM_txt_pos, self.state_tracker["COLOR"])
            else:
                self.state_tracker["yawn_start_time"] = time.perf_counter()
                self.state_tracker["YAWN_TIME"] = 0.0
                self.state_tracker["M_COLOR"] = self.GREEN
                self.state_tracker["play_mouth_alarm"] = False

            MOUTH_txt = f"MOUTH: {round(mouth_ratio,2)}"
            YAWN_TIME = f"YAWN: {round(self.state_tracker['YAWN_TIME'], 3)} Secs"
            plot_text(frame, MOUTH_txt, self.YAWN_on_txt_pos, self.state_tracker["M_COLOR"])
            plot_text(frame, YAWN_TIME, WARN_txt_pos, self.state_tracker["M_COLOR"])

        # 处理未检测到面部的情况
        else:
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["DROWSY_TIME"] = 0.0
            self.state_tracker["yawn_start_time"] = time.perf_counter()
            self.state_tracker["yawn_on"] = False
            self.state_tracker["YAWN_TIME"] = 0.0
            self.state_tracker["COLOR"] = self.GREEN
            self.state_tracker["M_COLOR"] = self.GREEN
            self.state_tracker["play_eye_alarm"] = False
            self.state_tracker["play_mouth_alarm"] = False
            # Flip the frame horizontally for a selfie-view display.
            frame = cv2.flip(frame, 1)

        return frame, self.state_tracker["play_eye_alarm"], self.state_tracker["play_mouth_alarm"]

