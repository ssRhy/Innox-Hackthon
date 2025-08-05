# /home/aaa/SkiroX/Agent/Agent.py (最终整合版)

import requests
import pygame
import time
import tempfile
import os
import pyaudio
from voice_recognition import VoiceRecognizer, get_silicon_response
from wake_word import WakeWordDetector

class VoiceChat:
    def __init__(self):
        # --- 密钥配置 ---
        self.BAIDU_API_KEY = "nkxC79rfnXZpGsa7p0LU4JxM"
        self.BAIDU_SECRET_KEY = "Xt2sg2gfAzbDWwAgZahSAaFCPcL0a5FT"
        self.PICOVOICE_ACCESS_KEY = "uJRC8ELPdAfQ5ol5ZgqxhpK/09X63nVn8wnG3Lnh48fNmei1Bv945A=="

        # --- 组件初始化 ---
        self.pyaudio_instance = None
        self.audio_stream = None
        self.wake_detector = WakeWordDetector(keywords=["jarvis"], access_key=self.PICOVOICE_ACCESS_KEY)
        self.voice_recognizer = VoiceRecognizer()
        self.baidu_access_token = None
        
        # 尝试初始化pygame，并捕获可能的错误
        try:
            print("正在初始化 Pygame Mixer...")
            pygame.mixer.init()
            print("Pygame Mixer 初始化成功。")
        except pygame.error as e:
            print(f"Pygame Mixer 初始化失败: {e}")
            print("请确保您的系统已正确安装声卡驱动。")
            pass

    def initialize_audio(self):
        """统一初始化PyAudio并打开一个共享的音频流"""
        try:
            print("  - 正在初始化 PyAudio...")
            self.pyaudio_instance = pyaudio.PyAudio()
            
            device_index = None
            print("  - 正在查找音频设备 'Yundea'...")
            for i in range(self.pyaudio_instance.get_device_count()):
                info = self.pyaudio_instance.get_device_info_by_index(i)
                if 'Yundea' in info['name']:
                    device_index = i
                    print(f"  - 已选择音频设备: {info['name']} (索引: {i})")
                    break
            
            if device_index is None:
                print("  - 警告: 未找到 'Yundea' USB麦克风, 将使用默认输入设备。")

            # 确保 porcupine 引擎已初始化
            if not self.wake_detector.porcupine:
                 print("  - 错误: Porcupine 引擎未初始化，无法获取帧长度。")
                 return False
            
            print("  - 正在打开共享音频流 (16000Hz)...")
            self.audio_stream = self.pyaudio_instance.open(
                rate=16000,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.wake_detector.porcupine.frame_length
            )

            print("  - 正在设置共享流到子模块...")
            self.wake_detector.setup_audio_stream(self.audio_stream)
            self.voice_recognizer.setup_audio_stream(self.audio_stream)
            
            return True
        except Exception as e:
            print(f"  - 音频初始化失败: {e}")
            return False

    def initialize(self):
        """初始化所有服务"""
        print("\n[开始初始化所有服务]")
        if not self.get_baidu_token(): return False
        if not self.wake_detector.initialize(): return False
        if not self.voice_recognizer.initialize(): return False
        if not self.initialize_audio(): return False
        print("[所有服务初始化完毕]\n")
        return True

    def get_baidu_token(self):
        print("1. 正在获取百度 Access Token...")
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.BAIDU_API_KEY}&client_secret={self.BAIDU_SECRET_KEY}"
        try:
            resp = requests.post(url)
            resp.raise_for_status()
            self.baidu_access_token = resp.json()['access_token']
            print("   ...百度 Token 获取成功。")
            return True
        except Exception as e:
            print(f"   - 获取百度 Token 失败: {e}")
            return False

    def text_to_speech(self, text):
        if not self.baidu_access_token or not text: return None
        url = "https://tsn.baidu.com/text2audio"
        payload = {'tex': text, 'tok': self.baidu_access_token, 'cuid': 'demo', 'ctp': 1, 'lan': 'zh', 'spd': 5, 'pit': 5, 'vol': 15, 'per': 4, 'aue': 3}
        try:
            resp = requests.post(url, data=payload)
            if resp.headers.get('Content-Type') == 'audio/mp3': return resp.content
        except Exception as e:
            print(f"语音合成请求失败: {e}")
        return None

    def play_audio(self, audio_data):
        if not audio_data or not pygame.mixer.get_init(): return
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                f.write(audio_data); path = f.name
            print("正在播放语音...")
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy(): time.sleep(0.1)
            print("语音播放完成！")
        except Exception as e:
            print(f"播放音频时发生错误: {e}")
        finally:
            if 'path' in locals() and os.path.exists(path): os.unlink(path)

    def chat_loop(self):
        print("=== 滑雪场智能助手启动 ===")
        print("我可以帮您：\n1. 提供滑雪场急救知识\n2. 查询实时天气\n3. 介绍雪道信息")
        try:
            while True:
                if not self.wake_detector.start_listening():
                    print("监听已停止。"); break
                user_input = self.voice_recognizer.start_recording()
                if not user_input or "退出" in user_input:
                    if "退出" in (user_input or ""): print("再见！"); break
                    continue
                ai_response = get_silicon_response(user_input)
                print(f"\nAI回复: {ai_response}")
                audio_data = self.text_to_speech(ai_response)
                self.play_audio(audio_data)
        except KeyboardInterrupt:
            print("\n程序已通过用户请求停止")

    def cleanup(self):
        print("\n正在清理所有资源...")
        if pygame.mixer.get_init(): pygame.mixer.quit()
        if self.audio_stream and self.audio_stream.is_active():
            self.audio_stream.stop_stream()
        if self.audio_stream: self.audio_stream.close()
        if self.pyaudio_instance: self.pyaudio_instance.terminate()
        if self.wake_detector: self.wake_detector.cleanup()
        if self.voice_recognizer: self.voice_recognizer.cleanup()
        print("清理完成。")

# 确保这部分代码在文件的最末尾，并且是顶格的
if __name__ == "__main__":
    print("--- Agent.py 开始执行 ---")
    chat = VoiceChat()
    if chat.initialize():
        try:
            chat.chat_loop()
        finally:
            chat.cleanup()
    else:
        print("\n[初始化失败，程序退出]")
        chat.cleanup()
    print("--- Agent.py 执行结束 ---")