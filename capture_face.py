import cv2
import time

def capture_face():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按空格键拍照，按ESC键退出")
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取画面")
            break
            
        # 显示画面
        cv2.imshow('Face Capture - Press SPACE to capture, ESC to exit', frame)
        
        # 等待按键
        key = cv2.waitKey(1)
        
        # 按空格键拍照
        if key == 32:  # 空格键的ASCII码
            # 保存图片
            cv2.imwrite('my_face.jpg', frame)
            print("照片已保存为 my_face.jpg")
            break
            
        # 按ESC键退出
        elif key == 27:  # ESC键的ASCII码
            print("已取消拍照")
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_face() 