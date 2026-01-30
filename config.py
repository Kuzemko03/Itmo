from dataclasses import dataclass
from typing import Optional, List, Dict
import json
from pathlib import Path

def load_secrets():
    secrets_path = Path(__file__).parent / "secrets.json"
    if secrets_path.exists():
        with open(secrets_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

_secrets = load_secrets()

@dataclass
class Config:
    GEMINI_API_KEY: str = _secrets.get("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-3-flash-preview"
    GEMINI_URL: str = "https://generativelanguage.googleapis.com/v1beta" 
    PROXY: Optional[str] = _secrets.get("PROXY")

    TEMPERATURE: float = 1.0 #советуют для 3.0 флеш
    MAX_TOKENS: int = 4096 #хватает

CFG = Config()

AVAILABLE_MODELS = {
    "Gemini 3 Flash (рекомендуется)": "gemini-3-flash-preview",
    "Gemini 3 Pro": "gemini-3-pro-preview",
    "Gemini 2 Flash": "gemini-2.0-flash",
}

STOP_WORDS = [
    "стоп", "stop", "хватит", "закончим", "завершить", "завершай",
    "фидбэк", "feedback", "достаточно", "конец", "давай фидбэк",
    "стоп игра", "стоп интервью", "заверши интервью"
]

DOCS_BY_TOPIC = {
    "python": "https://docs.python.org/3/tutorial/",
    "django": "https://docs.djangoproject.com/en/stable/",
    "flask": "https://flask.palletsprojects.com/",
    "fastapi": "https://fastapi.tiangolo.com/",
    "sql": "https://www.w3schools.com/sql/",
    "postgresql": "https://www.postgresql.org/docs/",
    "mysql": "https://dev.mysql.com/doc/",
    "git": "https://git-scm.com/book/ru/v2",
    "docker": "https://docs.docker.com/get-started/",
    "kubernetes": "https://kubernetes.io/ru/docs/",
    "javascript": "https://learn.javascript.ru/",
    "typescript": "https://www.typescriptlang.org/docs/",
    "react": "https://react.dev/learn",
    "vue": "https://ru.vuejs.org/guide/",
    "linux": "https://losst.pro/",
    "rest": "https://restfulapi.net/",
    "api": "https://restfulapi.net/",
    "oop": "https://realpython.com/python3-object-oriented-programming/",
    "ооп": "https://realpython.com/python3-object-oriented-programming/",
    "алгоритмы": "https://leetcode.com/",
    "algorithms": "https://leetcode.com/",
    "тестирование": "https://docs.pytest.org/",
    "pytest": "https://docs.pytest.org/",
    "asyncio": "https://docs.python.org/3/library/asyncio.html",
    "async": "https://docs.python.org/3/library/asyncio.html",
    "база данных": "https://www.w3schools.com/sql/",
    "базы данных": "https://www.w3schools.com/sql/",
    "database": "https://www.w3schools.com/sql/",
    "архитектур": "https://refactoring.guru/design-patterns",
    "паттерн": "https://refactoring.guru/design-patterns",
    "pattern": "https://refactoring.guru/design-patterns",
    "solid": "https://refactoring.guru/design-patterns",
    "проектирован": "https://refactoring.guru/design-patterns",
    "uml": "https://www.visual-paradigm.com/guide/uml-unified-modeling-language/",
    "диаграмм": "https://www.visual-paradigm.com/guide/uml-unified-modeling-language/",
    "redis": "https://redis.io/docs/",
    "celery": "https://docs.celeryq.dev/",
    "jwt": "https://jwt.io/introduction",
    "auth": "https://jwt.io/introduction",
    "orm": "https://docs.sqlalchemy.org/",
    "sqlalchemy": "https://docs.sqlalchemy.org/",
    "индекс": "https://use-the-index-luke.com/",
    "оптимизац": "https://use-the-index-luke.com/",
    "kafka": "https://kafka.apache.org/documentation/",
    "rabbitmq": "https://www.rabbitmq.com/tutorials",
    "очеред": "https://www.rabbitmq.com/tutorials",
    "ci/cd": "https://docs.github.com/en/actions",
    "ci cd": "https://docs.github.com/en/actions",
    "nginx": "https://nginx.org/ru/docs/",
    "http": "https://developer.mozilla.org/ru/docs/Web/HTTP",
}


def get_doc_url(topic: str) -> str:
    topic_lower = topic.lower()
    for key, url in DOCS_BY_TOPIC.items():
        if key in topic_lower:
            return url
    return "https://roadmap.sh/"

def get_multiple_resources(topic: str) -> List[str]:
    main_url = get_doc_url(topic)
    resources = [main_url]
    topic_lower = topic.lower()
    if "python" in topic_lower or "django" in topic_lower:
        resources.append("https://realpython.com/")
    if "sql" in topic_lower or "база" in topic_lower:
        resources.append("https://sqlbolt.com/")
    if "git" in topic_lower:
        resources.append("https://learngitbranching.js.org/")
    return resources[:3]

def adapt_log_to_tz_format(session_dict: Dict) -> Dict:
    """Адаптирует лог строго под формат ТЗ (только 3 поля в корне)"""
    
    # final_feedback — если объект, конвертим в строку
    fb = session_dict.get("final_feedback", "")
    if isinstance(fb, dict):
        # fallback если вдруг объект
        dec = fb.get("decision", {})
        fb = f"Грейд: {dec.get('evaluated_grade', 'N/A')}. Рекомендация: {dec.get('hiring_recommendation', 'N/A')}. {dec.get('explanation', '')}"
    
    # Строго 3 поля по ТЗ
    adapted = {
        "participant_name": session_dict.get("participant_name", ""),
        "turns": [],
        "final_feedback": fb if isinstance(fb, str) else ""
    }
    
    # Turns в порядке по ТЗ
    for turn in session_dict.get("turns", []):
        adapted["turns"].append({
            "turn_id": turn.get("turn_id"),
            "agent_visible_message": turn.get("agent_visible_message", ""),
            "user_message": turn.get("user_message", ""),
            "internal_thoughts": turn.get("internal_thoughts", "")
        })
    
    return adapted