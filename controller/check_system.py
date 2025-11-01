import sys
import subprocess

def check_python_version():
    version = sys.version_info
    print(f"🐍 Python версия: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Требуется Python 3.8 или выше!")
        return False
    print("✅ Версия Python подходит")
    return True

def check_dependencies():
    required = ['cv2', 'mediapipe', 'pyautogui']
    missing = []
    
    print("\n📦 Проверка зависимостей:")
    
    for package in required:
        try:
            if package == 'cv2':
                import cv2
                print(f"✅ OpenCV: {cv2.__version__}")
            elif package == 'mediapipe':
                import mediapipe
                print(f"✅ MediaPipe: {mediapipe.__version__}")
            elif package == 'pyautogui':
                import pyautogui
                print(f"✅ PyAutoGUI: {pyautogui.__version__}")
        except ImportError:
            print(f"❌ {package} не установлен")
            missing.append(package)
    
    if missing:
        print("\n⚠️  Установите недостающие пакеты:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_camera():

    print("\n📷 Проверка камеры:")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Камера недоступна")
            print("   Проверьте:")
            print("   - Подключена ли камера")
            print("   - Разрешения в Системных настройках → Камера")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print("✅ Камера работает")
            return True
        else:
            print("❌ Не удалось получить изображение с камеры")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при проверке камеры: {e}")
        return False

def check_macos():

    print("\n🍎 Проверка операционной системы:")
    
    if sys.platform != 'darwin':
        print(f"⚠️  Обнаружена ОС: {sys.platform}")
        print("   Этот скрипт оптимизирован для macOS")
        print("   Некоторые горячие клавиши могут не работать")
        return False
    
    print("✅ macOS обнаружена")
    return True

def print_accessibility_instructions():

    print("\n🔐 ВАЖНО - Настройка прав доступа:")
    print("=" * 50)
    print("Для управления MacBook нужны права Accessibility:")
    print()
    print("1. Системные настройки → Защита и безопасность")
    print("2. Конфиденциальность → Accessibility")
    print("3. Нажмите 🔒 и введите пароль")
    print("4. Добавьте Terminal (или вашу IDE)")
    print("5. Включите галочку")
    print("6. Перезапустите Terminal")
    print("=" * 50)

def main():
    print("=" * 50)
    print("🔍 ПРОВЕРКА СИСТЕМЫ")
    print("=" * 50)
    
    all_ok = True
    
    # Проверки
    all_ok = check_python_version() and all_ok
    all_ok = check_macos() and all_ok
    all_ok = check_dependencies() and all_ok
    all_ok = check_camera() and all_ok
    
    print("\n" + "=" * 50)
    
    if all_ok:
        print("✅ Все проверки пройдены!")
        print("\n🚀 Запустите программу:")
        print("   python3 gesture_control.py")
    else:
        print("⚠️  Есть проблемы, которые нужно исправить")
        print_accessibility_instructions()
    
    print("=" * 50)

if __name__ == "__main__":
    main()
