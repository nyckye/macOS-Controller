import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np
from collections import deque
import sys
import os

class UltimateAIController:
    def __init__(self):
        # MediaPipe инициализация
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_draw = mp.solutions.drawing_utils
        
        # Настройки
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
        # ЖЕСТЫ
        self.gesture_cooldown = 2.5
        self.last_action_time = 0
        self.stable_gesture = None
        self.stable_count = 0
        self.stability_threshold = 10
        
        # ГЛАЗА
        self.blink_history = deque(maxlen=100)
        self.eye_closed_start = None
        self.last_blink_time = 0
        self.blink_count = 0
        self.fatigue_threshold = 20
        self.long_blink_threshold = 1.5
        self.eye_action_cooldown = 3.0
        self.last_eye_action_time = 0
        
        # ГОЛОВА - ИСПРАВЛЕНО
        self.head_history = deque(maxlen=15)  # Больше истории для сглаживания
        self.head_action_cooldown = 3.0  # Увеличено до 3 секунд
        self.last_head_action_time = 0
        self.head_calibrated = False
        self.head_baseline_angle = 0.0
        self.head_baseline_y = 0.5
        
        # ОСАНКА - ИСПРАВЛЕНО
        self.posture_history = deque(maxlen=30)
        self.good_posture_baseline = None
        self.posture_check_interval = 3.0  # Увеличено до 3 секунд
        self.last_posture_check = 0
        self.slouch_warning_active = False
        self.slouch_count = 0
        self.good_posture_count = 0
        
        # Калибровка осанки - УЛУЧШЕНО
        self.calibration_mode = True
        self.calibration_frames = 0
        self.calibration_required = 90  # 3 секунды при 30 FPS
        self.calibration_data = []
        self.calibration_stage = "posture"  # "posture" или "head"
        
        # Счетчик кадров для стабильности
        self.frames_with_pose = 0
        self.frames_needed_for_detection = 5
        
        print("🚀 ПОЛНАЯ AI СИСТЕМА ЗАПУЩЕНА!")
        print("\n" + "="*60)
        print("📋 УПРАВЛЕНИЕ ЖЕСТАМИ:")
        print("="*60)
        print("   ✋ 5 пальцев → Новая вкладка")
        print("   ✊ Кулак → Закрыть вкладку")
        print("   ✌️  2 пальца → Следующая вкладка")
        print("   ☝️  1 палец → Предыдущая вкладка")
        print("   🤙 3 пальца → Обновить страницу")
        print("   👍 4 пальца → Mission Control")
        print("\n" + "="*60)
        print("👁️  УПРАВЛЕНИЕ ГЛАЗАМИ:")
        print("="*60)
        print("   😑 Закрыть глаза 1.5с → Пауза/Воспроизведение")
        print("   😴 Частое моргание → Уменьшить яркость")
        print("   👀 Редкое моргание → Увеличить яркость")
        print("\n" + "="*60)
        print("🧠 УПРАВЛЕНИЕ ГОЛОВОЙ:")
        print("="*60)
        print("   ⬅️  Наклон влево → Громкость ↓")
        print("   ➡️  Наклон вправо → Громкость ↑")
        print("   ⬆️  Наклон назад → Прокрутка вверх")
        print("   ⬇️  Наклон вперед → Прокрутка вниз")
        print("\n" + "="*60)
        print("🏃 КОНТРОЛЬ ОСАНКИ:")
        print("="*60)
        print("   🎯 СЯДЬТЕ РОВНО для калибровки!")
        print("   ⚠️  Система предупредит о сутулости")
        print("   ✅ Мониторинг шеи и плеч")
        print("\n⌨️  Нажмите 'q' для выхода\n")
    
    # ============= ЖЕСТЫ РУК =============
    
    def count_fingers_optimized(self, hand_landmarks):
        """Оптимизированный подсчет пальцев"""
        fingers = []
        landmarks = hand_landmarks.landmark
        
        wrist = landmarks[0]
        middle_base = landmarks[9]
        is_right_hand = wrist.x < middle_base.x
        
        # Большой палец
        thumb_tip = landmarks[4]
        thumb_base = landmarks[2]
        if is_right_hand:
            fingers.append(1 if thumb_tip.x < thumb_base.x else 0)
        else:
            fingers.append(1 if thumb_tip.x > thumb_base.x else 0)
        
        # Остальные пальцы
        finger_tips = [8, 12, 16, 20]
        finger_mids = [6, 10, 14, 18]
        
        for tip, mid in zip(finger_tips, finger_mids):
            if landmarks[tip].y < landmarks[mid].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return sum(fingers)
    
    def calculate_hand_confidence(self, hand_landmarks):
        """Проверка уверенности в жесте"""
        landmarks = hand_landmarks.landmark
        wrist = np.array([landmarks[0].x, landmarks[0].y])
        middle_finger = np.array([landmarks[9].x, landmarks[9].y])
        distance = np.linalg.norm(middle_finger - wrist)
        return 0.15 < distance < 0.4
    
    def get_gesture_name(self, finger_count):
        gestures = {
            0: "Кулак", 1: "Один палец", 2: "Два пальца",
            3: "Три пальца", 4: "Четыре пальца", 5: "Открытая ладонь"
        }
        return gestures.get(finger_count, "Неизвестно")
    
    def execute_gesture_action(self, finger_count):
        """Выполнение жеста"""
        current_time = time.time()
        
        if current_time - self.last_action_time < self.gesture_cooldown:
            return
        
        actions = {
            5: ("Новая вкладка", lambda: pyautogui.hotkey('command', 't')),
            0: ("Закрыть вкладку", lambda: pyautogui.hotkey('command', 'w')),
            2: ("Вкладка вправо", lambda: pyautogui.hotkey('command', 'shift', ']')),
            1: ("Вкладка влево", lambda: pyautogui.hotkey('command', 'shift', '[')),
            3: ("Обновить страницу", lambda: pyautogui.hotkey('command', 'r')),
            4: ("Mission Control", lambda: pyautogui.hotkey('ctrl', 'up'))
        }
        
        if finger_count in actions:
            action_name, action_func = actions[finger_count]
            try:
                action_func()
                self.last_action_time = current_time
                print(f"✅ Жест: {action_name}")
            except Exception as e:
                print(f"❌ Ошибка: {e}")
    
    # ============= ГЛАЗА =============
    
    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """EAR для моргания"""
        vertical1 = math.dist(
            [landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y],
            [landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y]
        )
        vertical2 = math.dist(
            [landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y],
            [landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]
        )
        horizontal = math.dist(
            [landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y],
            [landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y]
        )
        
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    
    def detect_eye_state(self, face_landmarks):
        """Состояние глаз"""
        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        landmarks = face_landmarks.landmark
        left_ear = self.calculate_eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = self.calculate_eye_aspect_ratio(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        
        return avg_ear < 0.2, avg_ear
    
    def handle_eye_actions(self, eyes_closed, current_time):
        """Действия глазами"""
        if current_time - self.last_eye_action_time < self.eye_action_cooldown:
            return
        
        if eyes_closed:
            if self.eye_closed_start is None:
                self.eye_closed_start = current_time
            else:
                closed_duration = current_time - self.eye_closed_start
                if closed_duration >= self.long_blink_threshold:
                    try:
                        pyautogui.press('space')
                        print("👁️  Глаза: Пауза/Воспроизведение")
                        self.last_eye_action_time = current_time
                        self.eye_closed_start = None
                    except:
                        pass
        else:
            if self.eye_closed_start is not None:
                closed_duration = current_time - self.eye_closed_start
                if closed_duration < self.long_blink_threshold:
                    self.blink_count += 1
                    self.blink_history.append(current_time)
                self.eye_closed_start = None
    
    def check_eye_fatigue(self, current_time):
        """Усталость глаз"""
        while self.blink_history and current_time - self.blink_history[0] > 60:
            self.blink_history.popleft()
        
        blinks_per_minute = len(self.blink_history)
        
        if blinks_per_minute > self.fatigue_threshold:
            if current_time - self.last_eye_action_time > 10:
                try:
                    import subprocess
                    subprocess.run(['osascript', '-e', 
                                  'tell application "System Events" to key code 107'], 
                                  capture_output=True, timeout=1)
                    print(f"😴 Усталость глаз ({blinks_per_minute} морг/мин) → Яркость ↓")
                    self.last_eye_action_time = current_time
                except:
                    pass
        
        return blinks_per_minute
    
    # ============= ГОЛОВА - ИСПРАВЛЕНО =============
    
    def calculate_head_pose(self, face_landmarks):
        """Вычисление позы головы"""
        landmarks = face_landmarks.landmark
        
        # Ключевые точки
        left_ear = np.array([landmarks[234].x, landmarks[234].y])
        right_ear = np.array([landmarks[454].x, landmarks[454].y])
        nose_tip = landmarks[1]
        
        # Угол наклона головы (лево/право)
        ear_angle = math.atan2(right_ear[1] - left_ear[1], right_ear[0] - left_ear[0])
        
        # Высота носа (вверх/вниз)
        nose_y = nose_tip.y
        
        return ear_angle, nose_y
    
    def calibrate_head(self):
        """Калибровка нейтрального положения головы"""
        if len(self.head_history) >= 15 and not self.head_calibrated:
            # Вычисляем среднее нейтральное положение
            angles = [h[0] for h in self.head_history]
            ys = [h[1] for h in self.head_history]
            
            self.head_baseline_angle = np.mean(angles)
            self.head_baseline_y = np.mean(ys)
            self.head_calibrated = True
            
            print(f"✅ Голова откалибрована! Базовый угол: {self.head_baseline_angle:.2f}, Y: {self.head_baseline_y:.2f}")
            return True
        return False
    
    def handle_head_actions(self, ear_angle, nose_y, current_time):
        """Управление головой - ИСПРАВЛЕНО"""
        if current_time - self.last_head_action_time < self.head_action_cooldown:
            return
        
        # Сохраняем историю
        self.head_history.append((ear_angle, nose_y))
        
        # Калибровка при первом запуске
        if not self.head_calibrated:
            self.calibrate_head()
            return
        
        if len(self.head_history) < 10:
            return
        
        # Вычисляем среднее за последние кадры
        recent_angles = [h[0] for h in list(self.head_history)[-10:]]
        recent_ys = [h[1] for h in list(self.head_history)[-10:]]
        
        avg_angle = np.mean(recent_angles)
        avg_y = np.mean(recent_ys)
        
        # Отклонение от базового положения
        angle_diff = avg_angle - self.head_baseline_angle
        y_diff = avg_y - self.head_baseline_y
        
        # Наклон влево/вправо (громкость) - УВЕЛИЧЕН ПОРОГ
        if angle_diff > 0.25:  # Было 0.15, стало 0.25
            try:
                import subprocess
                subprocess.run(['osascript', '-e', 
                              'set volume output volume ((output volume of (get volume settings)) + 10)'],
                              capture_output=True, timeout=1)
                print("➡️  Голова вправо: Громкость ↑")
                self.last_head_action_time = current_time
            except:
                pass
        elif angle_diff < -0.25:  # Было -0.15, стало -0.25
            try:
                import subprocess
                subprocess.run(['osascript', '-e', 
                              'set volume output volume ((output volume of (get volume settings)) - 10)'],
                              capture_output=True, timeout=1)
                print("⬅️  Голова влево: Громкость ↓")
                self.last_head_action_time = current_time
            except:
                pass
        # Наклон вперед/назад (прокрутка) - УВЕЛИЧЕН ПОРОГ
        elif y_diff > 0.08:  # Было 0.05, стало 0.08
            try:
                pyautogui.scroll(-3)
                print("⬇️  Голова вперед: Прокрутка вниз")
                self.last_head_action_time = current_time
            except:
                pass
        elif y_diff < -0.08:  # Было -0.05, стало -0.08
            try:
                pyautogui.scroll(3)
                print("⬆️  Голова назад: Прокрутка вверх")
                self.last_head_action_time = current_time
            except:
                pass
    
    # ============= ОСАНКА - ИСПРАВЛЕНО =============
    
    def calculate_posture_metrics(self, pose_landmarks):
        """Метрики осанки - УЛУЧШЕНО"""
        landmarks = pose_landmarks.landmark
        
        # Проверяем видимость ключевых точек
        visibility_threshold = 0.5
        required_points = [11, 12, 7, 8, 0]  # плечи, уши, нос
        
        for point in required_points:
            if landmarks[point].visibility < visibility_threshold:
                return None  # Недостаточно видимости
        
        # Ключевые точки
        left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
        right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
        left_ear = np.array([landmarks[7].x, landmarks[7].y, landmarks[7].z])
        right_ear = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
        nose = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        
        # Метрики
        shoulder_center = (left_shoulder + right_shoulder) / 2
        ear_center = (left_ear + right_ear) / 2
        
        # 1. Наклон плеч
        shoulder_angle = math.atan2(
            right_shoulder[1] - left_shoulder[1],
            right_shoulder[0] - left_shoulder[0]
        )
        
        # 2. Выдвижение шеи вперед
        neck_forward = abs(ear_center[0] - shoulder_center[0])
        
        # 3. Высота головы
        head_height = shoulder_center[1] - ear_center[1]
        
        # 4. Наклон головы вперед (Z координата)
        head_forward_z = ear_center[2] - shoulder_center[2]
        
        return {
            'shoulder_angle': abs(shoulder_angle),
            'neck_forward': neck_forward,
            'head_height': head_height,
            'head_forward_z': head_forward_z,
            'shoulder_y': shoulder_center[1],
            'ear_y': ear_center[1]
        }
    
    def calibrate_posture(self, metrics):
        """Калибровка осанки - УЛУЧШЕНО"""
        if self.calibration_mode and self.calibration_stage == "posture":
            if metrics is None:
                return False
            
            self.calibration_data.append(metrics)
            self.calibration_frames += 1
            
            if self.calibration_frames >= self.calibration_required:
                # Среднее значение правильной осанки
                self.good_posture_baseline = {
                    'shoulder_angle': np.median([m['shoulder_angle'] for m in self.calibration_data]),
                    'neck_forward': np.median([m['neck_forward'] for m in self.calibration_data]),
                    'head_height': np.median([m['head_height'] for m in self.calibration_data]),
                    'head_forward_z': np.median([m['head_forward_z'] for m in self.calibration_data]),
                    'shoulder_y': np.median([m['shoulder_y'] for m in self.calibration_data]),
                    'ear_y': np.median([m['ear_y'] for m in self.calibration_data])
                }
                
                self.calibration_mode = False
                print("\n✅ КАЛИБРОВКА ОСАНКИ ЗАВЕРШЕНА!")
                print(f"   Базовые значения:")
                print(f"   - Угол плеч: {self.good_posture_baseline['shoulder_angle']:.3f}")
                print(f"   - Шея вперед: {self.good_posture_baseline['neck_forward']:.3f}")
                print(f"   - Высота головы: {self.good_posture_baseline['head_height']:.3f}\n")
                return True
        return False
    
    def check_posture(self, metrics, current_time):
        """Проверка осанки - ИСПРАВЛЕНО"""
        if self.calibration_mode or self.good_posture_baseline is None or metrics is None:
            return None
        
        if current_time - self.last_posture_check < self.posture_check_interval:
            return None
        
        self.last_posture_check = current_time
        baseline = self.good_posture_baseline
        
        problems = []
        
        # УВЕЛИЧЕНЫ ПОРОГИ для меньших ложных срабатываний
        
        # 1. Наклон плеч - УВЕЛИЧЕН ПОРОГ
        if abs(metrics['shoulder_angle'] - baseline['shoulder_angle']) > 0.25:  # Было 0.15
            problems.append("ПЛЕЧИ НЕРОВНЫЕ")
        
        # 2. Шея вперед - УВЕЛИЧЕН ПОРОГ
        if metrics['neck_forward'] > baseline['neck_forward'] + 0.08:  # Было 0.05
            problems.append("ШЕЯ ВПЕРЕД")
            self.slouch_count += 1
        
        # 3. Голова опущена - УВЕЛИЧЕН ПОРОГ
        if metrics['head_height'] < baseline['head_height'] - 0.06:  # Было 0.03
            problems.append("ГОЛОВА ОПУЩЕНА")
            self.slouch_count += 1
        
        # 4. Плечи опущены - УВЕЛИЧЕН ПОРОГ
        if metrics['shoulder_y'] > baseline['shoulder_y'] + 0.08:  # Было 0.05
            problems.append("ПЛЕЧИ ОПУЩЕНЫ")
            self.slouch_count += 1
        
        if problems:
            self.slouch_warning_active = True
            return problems
        else:
            self.slouch_warning_active = False
            self.good_posture_count += 1
            if self.slouch_count > 0:
                self.slouch_count -= 1
            return []
    
    # ============= ОТРИСОВКА - РУССКИЙ ЯЗЫК =============
    
    def draw_info(self, frame, finger_count, gesture_name, blinks_per_min, 
                  ear_value, posture_problems):
        """Отображение информации НА РУССКОМ"""
        height, width = frame.shape[:2]
        
        # Основной блок
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 240), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_pos = 35
        
        # Жесты
        cv2.putText(frame, f"Жест: {gesture_name}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(frame, f"Пальцев: {finger_count}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        
        # Глаза
        cv2.putText(frame, f"Морганий/мин: {blinks_per_min}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += 25
        
        eye_status = "Закрыты" if ear_value < 0.2 else "Открыты"
        eye_color = (0, 0, 255) if ear_value < 0.2 else (0, 255, 0)
        cv2.putText(frame, f"Глаза: {eye_status}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
        y_pos += 30
        
        # Калибровка
        if self.calibration_mode:
            progress = int((self.calibration_frames / self.calibration_required) * 100)
            cv2.putText(frame, f"КАЛИБРОВКА: {progress}%", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_pos += 35
            cv2.putText(frame, "СИДИТЕ РОВНО!", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += 30
            cv2.putText(frame, "Спина прямая, плечи развернуты", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # ПРЕДУПРЕЖДЕНИЕ О СУТУЛОСТИ - БОЛЬШОЕ НА РУССКОМ
        if posture_problems and not self.calibration_mode and len(posture_problems) > 0:
            warning_overlay = frame.copy()
            box_width = 600
            box_height = 150 + len(posture_problems) * 40
            
            cv2.rectangle(warning_overlay, 
                         (width//2 - box_width//2, height//2 - box_height//2),
                         (width//2 + box_width//2, height//2 + box_height//2), 
                         (0, 0, 180), -1)
            cv2.addWeighted(warning_overlay, 0.85, frame, 0.15, 0, frame)
            
            # Заголовок
            cv2.putText(frame, "ВЫПРЯМИТЕСЬ!", 
                       (width//2 - 200, height//2 - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)
            
            # Проблемы
            y = height//2 + 10
            for problem in posture_problems:
                # Переводим на русский
                problem_ru = problem.replace("ШЕЯ ВПЕРЕД", "🔴 ШЕЯ ВЫТЯНУТА ВПЕРЕД") \
                                   .replace("ГОЛОВА ОПУЩЕНА", "🔴 ГОЛОВА СЛИШКОМ НИЗКО") \
                                   .replace("ПЛЕЧИ ОПУЩЕНЫ", "🔴 ПЛЕЧИ ОПУЩЕНЫ") \
                                   .replace("ПЛЕЧИ НЕРОВНЫЕ", "🔴 ПЛЕЧИ НА РАЗНОЙ ВЫСОТЕ")
                
                cv2.putText(frame, problem_ru, (width//2 - 250, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                y += 40
        
        # Счетчик сутулости
        if not self.calibration_mode:
            color = (0, 255, 0) if self.slouch_count < 5 else (0, 165, 255) if self.slouch_count < 10 else (0, 0, 255)
            cv2.putText(frame, f"Сутулость: {self.slouch_count}", (width - 250, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Инструкция
        cv2.putText(frame, "Нажмите 'q' для выхода", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    # ============= КАМЕРА - УЛУЧШЕНО ДЛЯ MACBOOK =============
    
    def find_camera(self):
        """Поиск камеры MacBook - УЛУЧШЕНО"""
        print("🔍 Поиск камеры MacBook...")
        
        # Для MacBook сначала пробуем AVFoundation
        print("   Попытка 1: AVFoundation (встроенная камера MacBook)...")
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("✅ Найдена встроенная камера MacBook!")
                return cap
            cap.release()
        
        # Пробуем стандартные индексы
        for i in [0, 1, 2]:
            print(f"   Попытка {i+2}: Камера {i}...")
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Камера найдена: индекс {i}")
                    return cap
                cap.release()
        
        return None
    
    # ============= ОСНОВНОЙ ЦИКЛ =============
    
    def run(self):
        """Основной цикл"""
        cap = self.find_camera()
        if cap is None:
            print("\n❌ КАМЕРА НЕ НАЙДЕНА!")
            print("\n🔧 Возможные решения:")
            print("   1. Закройте FaceTime, Zoom, Skype")
            print("   2. Проверьте: Системные настройки → Камера")
            print("   3. Перезагрузите MacBook")
            return
        
        # Настройка камеры
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n✅ СИСТЕМА АКТИВИРОВАНА!")
        print("🎯 СЯДЬТЕ РОВНО! Калибровка начнётся через 3 секунды...\n")
        
        # Задержка перед началом калибровки
        time.sleep(3)
        print("▶️  КАЛИБРОВКА НАЧАТА! Сидите ровно...\n")
        
        frame_count = 0
        error_count = 0
        
        try:
            while True:
                success, frame = cap.read()
                if not success or frame is None:
                    error_count += 1
                    if error_count >= 10:
                        print("❌ Слишком много ошибок. Камера недоступна.")
                        break
                    time.sleep(0.1)
                    continue
                
                error_count = 0
                frame_count += 1
                current_time = time.time()
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # РУКИ
                hand_results = self.hands.process(rgb_frame)
                finger_count = 0
                gesture_name = "Нет руки"
                
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        if self.calculate_hand_confidence(hand_landmarks):
                            self.mp_draw.draw_landmarks(
                                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                            
                            finger_count = self.count_fingers_optimized(hand_landmarks)
                            gesture_name = self.get_gesture_name(finger_count)
                            
                            if finger_count == self.stable_gesture:
                                self.stable_count += 1
                            else:
                                self.stable_gesture = finger_count
                                self.stable_count = 1
                            
                            if self.stable_count >= self.stability_threshold:
                                self.execute_gesture_action(finger_count)
                                self.stable_count = 0
                
                # ЛИЦО И ГЛАЗА
                face_results = self.face_mesh.process(rgb_frame)
                eyes_closed = False
                ear_value = 0.3
                
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        eyes_closed, ear_value = self.detect_eye_state(face_landmarks)
                        self.handle_eye_actions(eyes_closed, current_time)
                        
                        # Голова
                        ear_angle, nose_y = self.calculate_head_pose(face_landmarks)
                        self.handle_head_actions(ear_angle, nose_y, current_time)
                
                blinks_per_min = self.check_eye_fatigue(current_time)
                
                # ОСАНКА
                pose_results = self.pose.process(rgb_frame)
                posture_problems = None
                
                if pose_results.pose_landmarks:
                    self.frames_with_pose += 1
                    
                    if self.frames_with_pose >= self.frames_needed_for_detection:
                        metrics = self.calculate_posture_metrics(pose_results.pose_landmarks)
                        
                        if metrics is not None:
                            if self.calibration_mode:
                                self.calibrate_posture(metrics)
                            else:
                                posture_problems = self.check_posture(metrics, current_time)
                        
                        # Рисуем скелет (только ключевые точки)
                        # Плечи и голова
                        connections = [
                            (11, 12),  # Плечи
                            (7, 8),    # Уши
                            (0, 1),    # Нос
                        ]
                        for connection in connections:
                            start = pose_results.pose_landmarks.landmark[connection[0]]
                            end = pose_results.pose_landmarks.landmark[connection[1]]
                            
                            h, w, _ = frame.shape
                            start_point = (int(start.x * w), int(start.y * h))
                            end_point = (int(end.x * w), int(end.y * h))
                            
                            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
                            cv2.circle(frame, start_point, 5, (0, 255, 255), -1)
                            cv2.circle(frame, end_point, 5, (0, 255, 255), -1)
                
                # ОТРИСОВКА
                frame = self.draw_info(frame, finger_count, gesture_name,
                                      blinks_per_min, ear_value, posture_problems)
                
                cv2.imshow('AI Система Управления MacBook', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n👋 Выход из программы...")
                    break
                    
        except KeyboardInterrupt:
            print("\n👋 Программа остановлена пользователем")
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            self.face_mesh.close()
            self.pose.close()
            
            print(f"\n📊 СТАТИСТИКА СЕАНСА:")
            print(f"   ✅ Обработано кадров: {frame_count}")
            print(f"   👁️  Всего морганий: {self.blink_count}")
            print(f"   😴 Сутулость зафиксирована: {self.slouch_count} раз")
            print(f"   ✅ Хорошая осанка: {self.good_posture_count} проверок")
            
            if self.slouch_count > 0:
                print(f"\n⚠️  РЕКОМЕНДАЦИЯ: Вы сутулились {self.slouch_count} раз.")
                print("   Делайте перерывы каждые 30 минут!")
                print("   Следите за осанкой!")

def main():
    print("=" * 70)
    print("🤖 ПОЛНАЯ AI СИСТЕМА УПРАВЛЕНИЯ MACBOOK")
    print("   Жесты + Глаза + Голова + Контроль осанки")
    print("   Версия 2.0 - Улучшенная калибровка")
    print("=" * 70)
    
    controller = UltimateAIController()
    controller.run()

if __name__ == "__main__":
    main()
