import json
import re
import asyncio
import httpx
from typing import Dict, Optional
from config import Config, CFG, STOP_WORDS

class GeminiClient:
    def __init__(self, config: Config = CFG):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
    
    def set_model(self, model_id: str):
        self.config.GEMINI_MODEL = model_id
        # gemini-3 работает лучше с температурой 1.0, остальные с 0.7        
        self.config.TEMPERATURE = 1.0 if model_id.startswith("gemini-3") else 0.7
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            transport = httpx.AsyncHTTPTransport(proxy=self.config.PROXY) if self.config.PROXY else None
            self._client = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(60.0, connect=15.0))
            # 60 сек общий таймаут, 15 на коннект - эмпирически подобрано
        return self._client
    
    async def generate(self, prompt: str, temperature: float = None) -> str:
        client = await self._get_client()
        url = f"{self.config.GEMINI_URL}/models/{self.config.GEMINI_MODEL}:generateContent"
        
        temp = temperature if temperature is not None else self.config.TEMPERATURE
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temp, "maxOutputTokens": self.config.MAX_TOKENS}
        }
        headers = {"x-goog-api-key": self.config.GEMINI_API_KEY, "Content-Type": "application/json"}

        
        # retry при rate limit
        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                response = await client.post(url, json=payload, headers=headers)
                
                # 429 = rate limit, ждём и пробуем снова
                if response.status_code == 429:
                    wait = 2 + attempt * 2  # 2, 4, 6, 8 сек
                    print(f"Rate limit (429), жду {wait}с... (попытка {attempt + 1}/{max_attempts})")
                    await asyncio.sleep(wait)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if "candidates" in data and data["candidates"]:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                return ""
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_attempts - 1:
                    wait = 2 + attempt * 2
                    print(f"Rate limit (429), жду {wait}с... (попытка {attempt + 1}/{max_attempts})")
                    await asyncio.sleep(wait)
                    continue
                print(f"Ошибка LLM: {e}")
                return ""
            except Exception as e:
                print(f"Ошибка LLM: {e}")
                return ""
        
        print("Превышено число попыток LLM")
        return ""

    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

def parse_json_response(text: str) -> Optional[Dict]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    for pattern in [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```']:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
    
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None

async def is_stop_intent(llm: GeminiClient, message: str) -> bool:
    msg_lower = message.lower().strip()
    if any(word in msg_lower for word in STOP_WORDS):
        return True
    
    if len(message) < 100 and "?" not in message:
        prompt = f'''Определи, хочет ли пользователь ЯВНО ЗАВЕРШИТЬ интервью и получить фидбек.

Сообщение: "{message}"

ЗАВЕРШЕНИЕ (YES) — только если человек ПРЯМО просит закончить:
- "всё, хватит"
- "давай заканчивать"
- "устал, давай фидбек"
- "стоп, достаточно"
- "пока, завершай"

НЕ ЗАВЕРШЕНИЕ (NO) — любые ответы на вопросы интервью:
- "всё знаю" — это ОТВЕТ, не завершение
- "да я профессионал" — это ОТВЕТ
- "не знаю" — это ОТВЕТ
- "как погода" — это off-topic, но НЕ завершение
- любой технический ответ
- любая попытка ответить на вопрос
- хвастовство, грубость, глупости — это НЕ завершение

ВАЖНО: Если есть ЛЮБОЕ сомнение — отвечай NO.
Завершение только при ЯВНОМ намерении закончить интервью.

Ответь ТОЛЬКО: YES или NO'''
        response = await llm.generate(prompt, temperature=0.1)
        return response.strip().upper().startswith("YES")
    return False
