# /home/aaa/SkiroX/Agent/wake_word.py (最终版)

import pvporcupine
import struct

class WakeWordDetector:
    def __init__(self, keywords=["jarvis"], access_key=None):
        self.keywords, self.access_key = keywords, access_key
        self.porcupine, self.audio_stream = None, None

    def initialize(self):
        print("2. 正在初始化唤醒词引擎 (Porcupine)...")
        try:
            self.porcupine = pvporcupine.create(
                access_key=self.access_key, keywords=self.keywords, sensitivities=[0.5] * len(self.keywords))
            print("   ...唤醒词引擎初始化成功。")
            return True
        except pvporcupine.PorcupineError as e:
            print(f"   - Porcupine 初始化失败: {e}"); return False

    def setup_audio_stream(self, stream): self.audio_stream = stream

    def start_listening(self):
        if not self.porcupine or not self.audio_stream: return False
        print(f"\n正在监听唤醒词 '{self.keywords[0]}'...")
        try:
            while True:
                pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                if self.porcupine.process(pcm) >= 0:
                    print(f"检测到唤醒词: '{self.keywords[0]}'"); return True
        except (KeyboardInterrupt, IOError): return False

    def cleanup(self):
        if self.porcupine: self.porcupine.delete()
        print("Porcupine 引擎资源已清理。")