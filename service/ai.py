import re
from openai import OpenAI
from config import API_KEY, API_BASE_URL, API_MODEL


def clean_text(text):
    text = re.sub(r'[^\u4e00-\u9fa5\d\s，。、；：！？（）《》\-]', '', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def _handle_error(err, prefix="AI 请求失败"):
    e = str(err)
    if "timeout" in e.lower() or "timed out" in e.lower():
        return "AI 服务响应超时 请检查网络后重试"
    if "401" in e or "auth" in e.lower():
        return "API Key 无效 请检查配置"
    if "429" in e:
        return "API 请求频率超限 请稍后再试"
    return f"{prefix} {e}"


class AIService:
    def __init__(self):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE_URL,
            timeout=30.0,
        )
        self.model = API_MODEL

    def analyze(self, pest_type, soil_data=None, location=None, current_time=None, light_data=None):
        clean_type = pest_type.split(" - ", 1)[0].strip() if pest_type else pest_type

        prompt = f"病虫害：{clean_type}"
        if location:
            prompt += f"\n位置：{location}"
        if current_time:
            prompt += f"\n时间：{current_time}"
        if soil_data:
            prompt += f"\n土壤数据：{soil_data}"
        if light_data:
            prompt += f"\n光照数据：{light_data}"
        prompt += "\n\n简洁回答：病因、防治方法、预防要点。每项不超过3条。"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "农业病虫害防治专家。回答必须简洁，只说重点，不要废话。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            return clean_text(response.choices[0].message.content)
        except Exception as e:
            return _handle_error(e, "AI 分析失败")

    def chat(self, question, history=None):
        messages = [
            {"role": "system", "content": "农业病虫害防治专家。简洁回答，只说重点，不要废话。"}
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            return clean_text(response.choices[0].message.content)
        except Exception as e:
            return _handle_error(e, "AI 问答失败")


ai_service = AIService()