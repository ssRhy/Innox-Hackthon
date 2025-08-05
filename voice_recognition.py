# /home/aaa/SkiroX/Agent/voice_recognition.py (使用您提供的函数 + 诊断)

import dashscope
import time
import requests
import json

dashscope.api_key = "sk-9bffce1794ea4e85a1608d5617b258d3"

class VoiceRecognizer:
    # --- 这个类的所有代码都保持原样，无需修改 ---
    def __init__(self):
        self.audio_stream, self.callback, self.translator = None, None, None
        self.transcription_text, self.is_recording = "", False
        self.last_speech_time, self.last_text_length, self.silence_threshold = 0, 0, 1.5

    class RecognitionCallback(dashscope.audio.asr.TranslationRecognizerCallback):
        def __init__(self, parent): self.parent = parent
        def on_open(self): print("Dashscope 连接已建立。")
        def on_close(self): print("Dashscope 连接已关闭。")
        def on_event(self, id, tres, trres, usage):
            if tres and tres.text:
                self.parent.transcription_text = tres.text + (tres.stash.text if tres.stash else "")
                print(f"\r当前识别: {self.parent.transcription_text}", end="", flush=True)

    def initialize(self):
        print("3. 正在初始化语音识别器 (Dashscope)...")
        try:
            self.callback = self.RecognitionCallback(self)
            self.translator = dashscope.audio.asr.TranslationRecognizerRealtime(
                model="gummy-realtime-v1", format="pcm", sample_rate=16000,
                transcription_enabled=True, translation_enabled=False, callback=self.callback)
            print("   ...语音识别器初始化成功。")
            return True
        except Exception as e:
            print(f"   - Dashscope 初始化失败: {e}"); return False

    def setup_audio_stream(self, stream): self.audio_stream = stream

    def start_recording(self):
        if not self.audio_stream or not self.translator: return None
        print("\n开始录音..."); self.is_recording = True
        self.transcription_text, self.last_text_length = "", 0
        self.last_speech_time = time.time()
        try:
            self.translator.start()
            while self.is_recording:
                data = self.audio_stream.read(512 * 3, exception_on_overflow=False)
                self.translator.send_audio_frame(data)
                if len(self.transcription_text) > self.last_text_length:
                    self.last_speech_time = time.time()
                    self.last_text_length = len(self.transcription_text)
                if time.time() - self.last_speech_time > self.silence_threshold and self.last_text_length > 0:
                    print("\n检测到说话结束。"); break
        except Exception as e: print(f"录音错误: {e}")
        finally: self.stop_recording()
        time.sleep(0.5)
        final_text = self.transcription_text.strip()
        print(f"最终识别结果: {final_text}")
        return final_text

    def stop_recording(self):
        self.is_recording = False
        if self.translator:
            try: self.translator.stop()
            except dashscope.common.error.InvalidParameter: pass
    
    def cleanup(self):
        self.stop_recording()
        print("语音识别器清理完成。")

# --- 使用您提供的函数，并加入诊断代码 ---
def get_silicon_response(text):
    """调用硅基流动大模型获取回复"""
    if not text or not text.strip():
        return "未能识别到有效的语音输入"
    
    url = "https://api.siliconflow.cn/v1/chat/completions"
    api_key = "sk-rpbdfcumdlwdfssveutdyweficaabukciujbkoltmdsnjiba"
    
    system_prompt = """你是一个专业的滑雪助手，可以提供以下帮助：
1. 滑雪场急救知识：
   - 常见滑雪伤害的紧急处理方法
   - 高原反应的应对措施
   - 寒冷环境下的自救互救知识
   - 滑雪安全注意事项

2. 天气信息查询：
   - 实时天气状况
   - 降雪预报
   - 温度和风力情况
   - 雪场开放状态

3. 雪道信息：
   - 雪道难度等级说明
   - 雪道状况和积雪厚度
   - 缆车运行信息
   - 推荐适合的雪道路线

请根据用户的问题，提供准确、专业且易于理解的回答。在紧急情况下，优先提供安全建议。""" 
    
    payload = {
        "model": "Qwen/QwQ-32B",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    print(f"向AI模型 'Qwen/QwQ-32B' 发送: {text}")
    try:
        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()
        
        # --- 关键的诊断代码 ---
        print("--- 来自 AI 服务器的原始响应 ---")
        print(json.dumps(response_data, indent=2, ensure_ascii=False))
        print("-----------------------------")

        if 'choices' in response_data and len(response_data['choices']) > 0:
            return response_data['choices'][0]['message']['content']
        else:
            # 如果有错误信息，尝试返回更具体的错误
            if 'error' in response_data:
                return f"AI模型返回错误: {response_data['error'].get('message', '未知错误')}"
            return "抱歉，我没有得到有效的回复。"

    except Exception as e:
        print(f"调用AI接口时发生错误: {str(e)}")
        return "抱歉，发生了一些错误。"