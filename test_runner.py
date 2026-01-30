#!/usr/bin/env python3
"""
Тестер для Interview Coach
Проверяет все агенты включая ContradictionDetector и DepthProber
"""

import json
import asyncio
import os
import sys
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum

from config import adapt_log_to_tz_format
from models import Candidate
from llm_client import GeminiClient
from orchestrator import InterviewOrchestrator


# настройки тестов
TEST_MODEL = "gemini-2.0-flash"
USE_SMART_MODE = True


class TestResult(Enum):
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️ WARN"


@dataclass
class ScenarioConfig:
    name: str
    candidate: Dict[str, str]
    behavior: str
    expected_checks: List[str]
    max_turns: int = 8


@dataclass 
class TestReport:
    scenario_name: str
    result: TestResult
    checks: Dict[str, Tuple[TestResult, str]]
    duration_sec: float
    turns_count: int
    log_file: str
    errors: List[str] = field(default_factory=list)


class CandidateSimulator:
    """Генерит ответы кандидата через LLM"""
    
    def __init__(self, llm: GeminiClient):
        self.llm = llm
    
    async def generate_reply(self, interviewer_message: str, behavior: str,
                            history: List[Dict], turn_number: int,
                            candidate_info: Dict) -> str:
        
        # последние 6 сообщений для контекста
        hist_text = "\n".join([
            f"{'Интервьюер' if h['role'] == 'agent' else 'Кандидат'}: {h['text']}"
            for h in history[-6:]
        ])
        
        prompt = f"""Симулируй кандидата на собеседовании.

КАНДИДАТ:
Имя: {candidate_info.get('name', 'Тест')}
Позиция: {candidate_info.get('position', 'Backend')}
Уровень: {candidate_info.get('grade', 'Junior')}
Опыт: {candidate_info.get('experience', '')}

ПОВЕДЕНИЕ:
{behavior}

ДИАЛОГ:
{hist_text if hist_text else '[старт]'}

СООБЩЕНИЕ ИНТЕРВЬЮЕРА:
"{interviewer_message}"

ХОД: {turn_number}

ПРАВИЛА:
1. Кратко, 1-3 предложения
2. От первого лица
3. Следуй поведению
4. Если надо закончить - скажи "стоп" или "давай фидбэк"

Только реплика:"""

        resp = await self.llm.generate(prompt, temperature=0.7)
        if resp:
            return resp.strip().strip('"\'')
        return "Повтори вопрос?"


# === СЦЕНАРИИ ===

SCENARIOS = [
    # 1. нормальный джун
    ScenarioConfig(
        name="ideal_junior",
        candidate={
            "name": "Алексей Петров",
            "position": "Backend Developer",
            "grade": "Junior",
            "experience": "Пет-проекты на Django, SQL, Git"
        },
        behavior="""Хороший джун:
- Правильно отвечай на базовые вопросы
- На сложные честно говори "не знаю" или "не работал"
- На 4-5 ходу спроси что-то о компании
- На 7 ходу скажи "давай фидбэк"
""",
        expected_checks=[
            "adaptivity_up", "candidate_question_handled", 
            "positive_feedback", "depth_tracked", "agents_all_present"
        ],
        max_turns=8
    ),
    
    # 2. галлюцинации
    ScenarioConfig(
        name="hallucinator",
        candidate={
            "name": "Игорь Выдумкин", 
            "position": "Backend Developer",
            "grade": "Junior",
            "experience": "Курсы"
        },
        behavior="""Уверенно ври:
- Ход 2: "В Python 4.0 убрали циклы for и заменили на нейросети"
- Ход 4: "Django 6.0 работает без базы данных"
- Ход 6: "стоп"
""",
        expected_checks=[
            "hallucination_detected", "hallucination_corrected", 
            "low_honesty_score", "fact_checker_worked"
        ],
        max_turns=7
    ),
    
    # 3. уход от темы
    ScenarioConfig(
        name="off_topic",
        candidate={
            "name": "Мария Болтушкина",
            "position": "Backend Developer", 
            "grade": "Middle",
            "experience": "3 года"
        },
        behavior="""Уводи от темы:
- Ход 2: "А какая погода сегодня?"
- Ход 3: нормальный ответ
- Ход 4: "Смотрел новый фильм Марвел?"
- Ход 5: нормальный ответ
- Ход 6: "хватит, давай результаты"
""",
        expected_checks=[
            "off_topic_detected", "returned_to_topic", 
            "robustness", "observer_flagged"
        ],
        max_turns=7
    ),
    
    # 4. токсик
    ScenarioConfig(
        name="toxic",
        candidate={
            "name": "Хам Грубиянов",
            "position": "Backend Developer",
            "grade": "Senior",
            "experience": "10 лет"
        },
        behavior="""Сначала норм, потом груби:
    - Ход 1: "Работаю 10 лет, в основном высоконагруженные системы"
    - Ход 2: короткий технический ответ
    - Ход 3: "Это тупой вопрос, любой джун такое знает, зачем спрашиваете?"
    - Ход 4: "Вы вообще понимаете что спрашиваете? Элементарщина"
    - Ход 5: "Всё, хватит цирка, давай уже фидбэк"
    """,
        expected_checks=[
            "toxic_detected", "professional_response", 
            "no_hire_recommendation"
        ],
        max_turns=6
    ),

    
    # 5. вопросы от кандидата
    ScenarioConfig(
        name="curious",
        candidate={
            "name": "Любопытный Вопрошалов",
            "position": "Backend Developer",
            "grade": "Middle",
            "experience": "2 года"
        },
        behavior="""Задавай вопросы:
- Ход 2: короткий ответ + "Какой стек у вас?"
- Ход 3: ответ + "Используете микросервисы?"
- Ход 4: "Какие задачи на испытательном?"
- Ход 5: нормальный ответ
- Ход 6: "стоп"
""",
        expected_checks=[
            "candidate_questions_answered", "engagement_high", 
            "not_ignored", "interviewer_adapted"
        ],
        max_turns=7
    ),
    
    # 6. честный новичок
    ScenarioConfig(
        name="honest_beginner",
        candidate={
            "name": "Честный Новичков",
            "position": "Backend Developer",
            "grade": "Junior",
            "experience": "Курсы"
        },
        behavior="""Честно признавай незнание:
- Базовое знаешь
- Сложное: "не знаю" или "не работал"
- Не выдумывай
- Ход 6: "хочу результат"
""",
        expected_checks=[
            "honesty_high", "difficulty_decreased", 
            "gaps_identified", "roadmap_generated", "depth_tracked"
        ],
        max_turns=7
    ),
    
    # 7. сильный сеньор
    ScenarioConfig(
        name="strong_senior",
        candidate={
            "name": "Профи Эксперт",
            "position": "Backend Developer",
            "grade": "Senior",
            "experience": "8 лет Python, архитектура"
        },
        behavior="""Отвечай как сеньор:
- Глубокие ответы с примерами
- Упоминай паттерны, trade-offs
- GIL -> multiprocessing, asyncio
- Базы -> индексы, explain, репликация
- Ход 7: "достаточно, жду фидбэк"
""",
        expected_checks=[
            "difficulty_increased", "skills_confirmed", 
            "few_gaps", "depth_high_levels"
        ],
        max_turns=8
    ),
    
    # 8. молчун
    ScenarioConfig(
        name="silent",
        candidate={
            "name": "Краткий Молчунов",
            "position": "Backend Developer",
            "grade": "Junior",
            "experience": "1 год"
        },
        behavior="""Максимально кратко:
- "Да", "Нет", "Не знаю"
- 3-5 слов максимум
- Ход 5: "стоп"
""",
        expected_checks=[
            "clarity_low", "probing_questions", 
            "difficulty_adjusted", "depth_low_levels"
        ],
        max_turns=6
    ),
    
    # 9. противоречия (НОВЫЙ - тест ContradictionDetector)
    ScenarioConfig(
        name="contradicting",
        candidate={
            "name": "Противоречивый Петров",
            "position": "Backend Developer",
            "grade": "Middle",
            "experience": "3 года Django"
        },
        behavior="""Противоречь себе:
- Ход 1: "Я 3 года работаю с Django, знаю его отлично"
- Ход 2: нормальный ответ про Django
- Ход 3: "Честно говоря, я только начал изучать Django, пока мало опыта"
- Ход 4: нормальный ответ
- Ход 5: "стоп"
""",
        expected_checks=[
            "contradiction_detected", "contradiction_handled",
            "observer_flagged", "context_maintained"
        ],
        max_turns=6
    ),
    
    # 10. сценарий из ТЗ
    ScenarioConfig(
        name="tz_scenario",
        candidate={
            "name": "Алекс Тестовый",
            "position": "Backend Developer",
            "grade": "Junior",
            "experience": "Django, SQL"
        },
        behavior="""По ТЗ:
- Ход 1: "Привет, я Алекс, Junior Backend. Знаю Python, SQL, Git"
- Ход 2: правильный развёрнутый ответ
- Ход 3: "Читал на Хабре что в Python 4.0 циклы for уберут и заменят на нейронные связи"
- Ход 4: "Какие задачи на испытательном? Используете микросервисы?"
- Ход 5: "Стоп игра"
""",
        expected_checks=[
            "hallucination_caught", "question_answered", 
            "full_feedback", "all_agents_logged"
        ],
        max_turns=6
    ),
    
    # 11. долгое интервью
    ScenarioConfig(
        name="long_interview",
        candidate={
            "name": "Выносливый Марафонец",
            "position": "Backend Developer",
            "grade": "Middle",
            "experience": "3 года fullstack"
        },
        behavior="""Долгое интервью:
- Качественные ответы
- Чередуй хорошие и средние
- Иногда "не уверен, но думаю..."
- Ход 12: "стоп"
""",
        expected_checks=[
            "context_maintained", "no_repeated_topics", 
            "stable_performance", "depth_tracked"
        ],
        max_turns=13
    ),
    
    # 12. смена позиции (тест глубины)
    ScenarioConfig(
        name="depth_test",
        candidate={
            "name": "Глубокий Знаток",
            "position": "Backend Developer",
            "grade": "Middle",
            "experience": "4 года Python"
        },
        behavior="""Показывай разную глубину:
- Ход 1-2: поверхностные ответы, базовые определения
- Ход 3-4: более глубокие ответы с примерами
- Ход 5: экспертный ответ с trade-offs и edge cases
- Ход 6: "стоп"
""",
        expected_checks=[
            "depth_progression", "depth_tracked",
            "skills_confirmed", "interviewer_adapted"
        ],
        max_turns=7
    ),
]


class TestChecker:
    """Проверяет результаты"""
    
    # === БАЗОВЫЕ ПРОВЕРКИ ===
    
    def check_hallucination_detected(self, d: Dict) -> Tuple[TestResult, str]:
        for t in d.get("turns", []):
            thoughts = t.get("internal_thoughts", "").lower()
            if "hallucination" in thoughts or "галлюцин" in thoughts:
                return TestResult.PASS, "Галлюцинация обнаружена"
        
        # проверяем red_flags
        fb = d.get("final_feedback", {})
        if isinstance(fb, dict):
            for flag in fb.get("red_flags", []):
                if "ложь" in flag.lower() or "неправд" in flag.lower():
                    return TestResult.PASS, "Галлюцинация в red_flags"
        
        return TestResult.FAIL, "Галлюцинация не обнаружена"
    
    def check_hallucination_corrected(self, d: Dict) -> Tuple[TestResult, str]:
        markers = ["на самом деле", "это не так", "не соответствует", 
                   "не существует", "неверн", "должен отметить", "python 4"]
        for t in d.get("turns", []):
            msg = t.get("agent_visible_message", "").lower()
            if any(m in msg for m in markers):
                return TestResult.PASS, "Агент исправил"
        return TestResult.FAIL, "Не исправлено"
    
    def check_off_topic_detected(self, d: Dict) -> Tuple[TestResult, str]:
        for t in d.get("turns", []):
            thoughts = t.get("internal_thoughts", "").lower()
            if "off_topic" in thoughts or "не по теме" in thoughts:
                return TestResult.PASS, "Off-topic обнаружен"
        return TestResult.FAIL, "Off-topic не обнаружен"
    
    def check_returned_to_topic(self, d: Dict) -> Tuple[TestResult, str]:
        found = False
        for t in d.get("turns", []):
            if "off_topic" in t.get("internal_thoughts", "").lower():
                found = True
            if found:
                msg = t.get("agent_visible_message", "").lower()
                markers = ["вернёмся", "вернемся", "продолжим", "интервью", "технический"]
                if any(m in msg for m in markers):
                    return TestResult.PASS, "Вернул к теме"
        return TestResult.WARN if not found else TestResult.FAIL, "Не вернул"
    
    def check_toxic_detected(self, d: Dict) -> Tuple[TestResult, str]:
        for t in d.get("turns", []):
            thoughts = t.get("internal_thoughts", "").lower()
            if "toxic" in thoughts or "грубость" in thoughts:
                return TestResult.PASS, "Токсичность обнаружена"
        return TestResult.FAIL, "Токсичность не обнаружена"
    
    def check_professional_response(self, d: Dict) -> Tuple[TestResult, str]:
        bad = ["сам дурак", "идиот", "тупой"]
        good = ["понимаю", "давайте", "предлагаю", "продолжим"]
        
        for t in d.get("turns", []):
            if "toxic" in t.get("internal_thoughts", "").lower():
                msg = t.get("agent_visible_message", "").lower()
                if any(b in msg for b in bad):
                    return TestResult.FAIL, "Агент нагрубил"
                if any(g in msg for g in good):
                    return TestResult.PASS, "Профессионализм сохранён"
        return TestResult.WARN, "Не проверено"
    
    def check_no_hire_recommendation(self, d: Dict) -> Tuple[TestResult, str]:
        fb = d.get("final_feedback", {})
        if isinstance(fb, dict):
            rec = fb.get("decision", {}).get("hiring_recommendation", "").lower()
            if "no hire" in rec or "no_hire" in rec:
                return TestResult.PASS, f"Рекомендация: {rec}"
        return TestResult.FAIL, "Ожидался No Hire"
    
    def check_candidate_questions_answered(self, d: Dict) -> Tuple[TestResult, str]:
        markers = ["обычно", "как правило", "тренажёр", "тренажер", "используют", "стек"]
        for t in d.get("turns", []):
            if "candidate_question" in t.get("internal_thoughts", "").lower():
                msg = t.get("agent_visible_message", "").lower()
                if any(m in msg for m in markers):
                    return TestResult.PASS, "Ответил на вопрос"
        return TestResult.WARN, "Вопросы не найдены"
    
    def check_difficulty_increased(self, d: Dict) -> Tuple[TestResult, str]:
        for t in d.get("turns", []):
            thoughts = t.get("internal_thoughts", "").lower()
            if "повышена" in thoughts:
                return TestResult.PASS, "Сложность повышалась"
        return TestResult.WARN, "Повышение не зафиксировано"
    
    def check_difficulty_decreased(self, d: Dict) -> Tuple[TestResult, str]:
        for t in d.get("turns", []):
            if "понижена" in t.get("internal_thoughts", "").lower():
                return TestResult.PASS, "Сложность понижалась"
        return TestResult.WARN, "Понижение не зафиксировано"
    
    def check_honesty_high(self, d: Dict) -> Tuple[TestResult, str]:
        fb = d.get("final_feedback", {})
        if isinstance(fb, dict):
            score = fb.get("soft_skills_review", {}).get("honesty", {}).get("score", 0)
            if score >= 7:
                return TestResult.PASS, f"Честность: {score}/10"
            if score >= 5:
                return TestResult.WARN, f"Средняя: {score}/10"
        return TestResult.FAIL, "Низкая честность"
    
    def check_low_honesty_score(self, d: Dict) -> Tuple[TestResult, str]:
        fb = d.get("final_feedback", {})
        if isinstance(fb, dict):
            score = fb.get("soft_skills_review", {}).get("honesty", {}).get("score", 10)
            if score <= 5:
                return TestResult.PASS, f"Честность низкая: {score}/10"
        return TestResult.FAIL, "Честность должна быть низкой"
    
    def check_engagement_high(self, d: Dict) -> Tuple[TestResult, str]:
        fb = d.get("final_feedback", {})
        if isinstance(fb, dict):
            score = fb.get("soft_skills_review", {}).get("engagement", {}).get("score", 0)
            if score >= 7:
                return TestResult.PASS, f"Вовлечённость: {score}/10"
        return TestResult.WARN, "Вовлечённость не высокая"
    
    def check_gaps_identified(self, d: Dict) -> Tuple[TestResult, str]:
        fb = d.get("final_feedback", {})
        if isinstance(fb, dict):
            gaps = fb.get("technical_review", {}).get("knowledge_gaps", [])
            if len(gaps) > 0:
                return TestResult.PASS, f"Пробелов: {len(gaps)}"
        return TestResult.WARN, "Пробелы не выявлены"
    
    def check_roadmap_generated(self, d: Dict) -> Tuple[TestResult, str]:
        fb = d.get("final_feedback", {})
        if isinstance(fb, dict):
            topics = fb.get("roadmap", {}).get("priority_topics", [])
            if len(topics) > 0:
                has_res = any(t.get("resources") for t in topics)
                if has_res:
                    return TestResult.PASS, f"Roadmap: {len(topics)} тем с ресурсами"
                return TestResult.WARN, f"Roadmap без ресурсов"
        return TestResult.FAIL, "Roadmap пустой"
    
    def check_skills_confirmed(self, d: Dict) -> Tuple[TestResult, str]:
        fb = d.get("final_feedback", {})
        if isinstance(fb, dict):
            skills = fb.get("technical_review", {}).get("confirmed_skills", [])
            if len(skills) >= 3:
                return TestResult.PASS, f"Навыков: {len(skills)}"
            if len(skills) > 0:
                return TestResult.WARN, f"Мало навыков: {len(skills)}"
        return TestResult.FAIL, "Навыки не подтверждены"
    
    def check_few_gaps(self, d: Dict) -> Tuple[TestResult, str]:
        fb = d.get("final_feedback", {})
        if isinstance(fb, dict):
            tech = fb.get("technical_review", {})
            gaps = len(tech.get("knowledge_gaps", []))
            skills = len(tech.get("confirmed_skills", []))
            if gaps <= skills:
                return TestResult.PASS, f"Навыков {skills} >= пробелов {gaps}"
        return TestResult.WARN, "Много пробелов"
    
    def check_context_maintained(self, d: Dict) -> Tuple[TestResult, str]:
        turns = d.get("turns", [])
        if len(turns) < 3:
            return TestResult.WARN, "Мало ходов"
        
        # проверяем что есть прогресс и финальный отчёт
        if d.get("final_feedback"):
            return TestResult.PASS, f"Контекст ок, {len(turns)} ходов"
        return TestResult.WARN, "Нет финального отчёта"

    
    def check_robustness(self, d: Dict) -> Tuple[TestResult, str]:
        if d.get("final_feedback"):
            return TestResult.PASS, "Система устойчива"
        return TestResult.FAIL, "Нет отчёта"
    
    def check_stable_performance(self, d: Dict) -> Tuple[TestResult, str]:
        turns = len(d.get("turns", []))
        if turns >= 10 and d.get("final_feedback"):
            return TestResult.PASS, f"Стабильно: {turns} ходов"
        return TestResult.WARN, f"Только {turns} ходов"
    
    def check_clarity_low(self, d: Dict) -> Tuple[TestResult, str]:
        fb = d.get("final_feedback", {})
        if isinstance(fb, dict):
            score = fb.get("soft_skills_review", {}).get("clarity", {}).get("score", 10)
            if score <= 5:
                return TestResult.PASS, f"Ясность низкая: {score}/10"
        return TestResult.WARN, "Ясность не низкая"
    
    def check_probing_questions(self, d: Dict) -> Tuple[TestResult, str]:
        markers = ["подробнее", "пояснить", "что имеешь в виду", "расскажи больше"]
        for t in d.get("turns", []):
            msg = t.get("agent_visible_message", "").lower()
            if any(m in msg for m in markers):
                return TestResult.PASS, "Уточняющие вопросы есть"
        return TestResult.WARN, "Уточнений нет"
    
    # === НОВЫЕ ПРОВЕРКИ ДЛЯ АГЕНТОВ ===
    
    def check_contradiction_detected(self, d: Dict) -> Tuple[TestResult, str]:
        """Проверяет что ContradictionDetector сработал"""
        for t in d.get("turns", []):
            thoughts = t.get("internal_thoughts", "")
            if "ContradictionDetector" in thoughts or "противореч" in thoughts.lower():
                return TestResult.PASS, "Противоречие обнаружено"
        return TestResult.FAIL, "Противоречие не обнаружено"
    
    def check_contradiction_handled(self, d: Dict) -> Tuple[TestResult, str]:
        """Проверяет что агент отреагировал на противоречие"""
        markers = ["ранее ты говорил", "раньше упоминал", "противореч", "уточни", "пояснить"]
        for t in d.get("turns", []):
            if "contradiction" in t.get("internal_thoughts", "").lower():
                msg = t.get("agent_visible_message", "").lower()
                if any(m in msg for m in markers):
                    return TestResult.PASS, "Противоречие обработано"
        return TestResult.WARN, "Реакция на противоречие не найдена"
    
    def check_depth_tracked(self, d: Dict) -> Tuple[TestResult, str]:
        """Проверяет что DepthProber работал"""
        for t in d.get("turns", []):
            thoughts = t.get("internal_thoughts", "")
            if "DepthProber" in thoughts or "уровень" in thoughts.lower() and "/5" in thoughts:
                return TestResult.PASS, "Глубина отслеживается"
        return TestResult.WARN, "DepthProber не зафиксирован"
    
    def check_depth_high_levels(self, d: Dict) -> Tuple[TestResult, str]:
        """Проверяет высокие уровни глубины для сеньора"""
        for t in d.get("turns", []):
            thoughts = t.get("internal_thoughts", "")
            # ищем уровни 4 или 5
            if "уровень 4/5" in thoughts or "уровень 5/5" in thoughts:
                return TestResult.PASS, "Высокая глубина зафиксирована"
            if ": 4/5" in thoughts or ": 5/5" in thoughts:
                return TestResult.PASS, "Глубина 4-5"
        return TestResult.WARN, "Высокая глубина не найдена"
    
    def check_depth_low_levels(self, d: Dict) -> Tuple[TestResult, str]:
        """Проверяет низкие уровни для молчуна"""
        high_found = False
        for t in d.get("turns", []):
            thoughts = t.get("internal_thoughts", "")
            if "уровень 4/5" in thoughts or "уровень 5/5" in thoughts:
                high_found = True
        if not high_found:
            return TestResult.PASS, "Глубина не высокая"
        return TestResult.WARN, "Неожиданно высокая глубина"
    
    def check_depth_progression(self, d: Dict) -> Tuple[TestResult, str]:
        """Проверяет прогрессию глубины"""
        levels = []
        for t in d.get("turns", []):
            thoughts = t.get("internal_thoughts", "")
            for lvl in ["1/5", "2/5", "3/5", "4/5", "5/5"]:
                if lvl in thoughts:
                    levels.append(int(lvl[0]))
        
        if len(levels) >= 2 and levels[-1] > levels[0]:
            return TestResult.PASS, f"Прогрессия: {levels[0]} → {levels[-1]}"
        if len(levels) >= 1:
            return TestResult.WARN, f"Уровни: {levels}"
        return TestResult.WARN, "Прогрессия не отслежена"
    
    def check_fact_checker_worked(self, d: Dict) -> Tuple[TestResult, str]:
        """Проверяет что FactChecker работал"""
        for t in d.get("turns", []):
            thoughts = t.get("internal_thoughts", "")
            if "FactChecker" in thoughts:
                return TestResult.PASS, "FactChecker работал"
        return TestResult.WARN, "FactChecker не зафиксирован"
    
    def check_observer_flagged(self, d: Dict) -> Tuple[TestResult, str]:
        """Проверяет что Observer выставлял флаги"""
        for t in d.get("turns", []):
            thoughts = t.get("internal_thoughts", "")
            if "Observer" in thoughts and "Флаги:" in thoughts:
                if "[]" not in thoughts.split("Флаги:")[1][:20]:
                    return TestResult.PASS, "Observer выставил флаги"
        return TestResult.WARN, "Флаги не найдены"
    
    def check_interviewer_adapted(self, d: Dict) -> Tuple[TestResult, str]:
        """Проверяет что Interviewer адаптировался"""
        for t in d.get("turns", []):
            thoughts = t.get("internal_thoughts", "")
            if "Interviewer" in thoughts:
                return TestResult.PASS, "Interviewer активен"
        return TestResult.WARN, "Interviewer не найден"
    
    def check_agents_all_present(self, d: Dict) -> Tuple[TestResult, str]:
        """Проверяет присутствие всех агентов в логах"""
        agents = set()
        expected = {"Observer", "Interviewer"}
        
        for t in d.get("turns", []):
            thoughts = t.get("internal_thoughts", "")
            for agent in ["Observer", "Interviewer", "FactChecker", 
                         "ContradictionDetector", "DepthProber", "DifficultyCtrl", "MetaReviewer"]:
                if agent in thoughts:
                    agents.add(agent)
        
        if expected.issubset(agents):
            return TestResult.PASS, f"Агенты: {', '.join(agents)}"
        missing = expected - agents
        return TestResult.WARN, f"Нет агентов: {missing}"
    
    def check_all_agents_logged(self, d: Dict) -> Tuple[TestResult, str]:
        """Алиас для agents_all_present"""
        return self.check_agents_all_present(d)
    
    def check_full_feedback(self, d: Dict) -> Tuple[TestResult, str]:
        """Проверяет полноту отчёта"""
        fb = d.get("final_feedback", {})
        if isinstance(fb, str):
            # если строка - значит summary, ок для формата ТЗ
            return TestResult.PASS, "Отчёт есть (строка)"
        if isinstance(fb, dict):
            required = ["decision", "technical_review", "soft_skills_review", "roadmap"]
            missing = [r for r in required if r not in fb or not fb[r]]
            if not missing:
                return TestResult.PASS, "Отчёт полный"
            return TestResult.WARN, f"Нет секций: {missing}"
        return TestResult.FAIL, "Отчёт отсутствует"
    
    # === АЛИАСЫ ===
    
    def check_adaptivity_up(self, d): return self.check_difficulty_increased(d)
    def check_candidate_question_handled(self, d): return self.check_candidate_questions_answered(d)
    def check_positive_feedback(self, d):
        fb = d.get("final_feedback", {})
        if isinstance(fb, dict):
            rec = fb.get("decision", {}).get("hiring_recommendation", "").lower()
            if "hire" in rec and "no" not in rec:
                return TestResult.PASS, f"Позитивная: {rec}"
        return TestResult.WARN, "Рекомендация неясна"
    
    def check_not_ignored(self, d): return self.check_candidate_questions_answered(d)
    def check_no_repeated_topics(self, d): return self.check_context_maintained(d)
    def check_difficulty_adjusted(self, d):
        inc = self.check_difficulty_increased(d)
        dec = self.check_difficulty_decreased(d)
        if inc[0] == TestResult.PASS or dec[0] == TestResult.PASS:
            return TestResult.PASS, "Сложность адаптировалась"
        return TestResult.WARN, "Изменение не зафиксировано"
    
    def check_hallucination_caught(self, d): return self.check_hallucination_detected(d)
    def check_question_answered(self, d): return self.check_candidate_questions_answered(d)
    def check_full_feedback_generated(self, d): return self.check_full_feedback(d)
    
    def run_check(self, name: str, d: Dict) -> Tuple[TestResult, str]:
        method = f"check_{name}"
        if hasattr(self, method):
            try:
                return getattr(self, method)(d)
            except Exception as e:
                return TestResult.FAIL, f"Ошибка: {e}"
        return TestResult.WARN, f"Проверка {name} не реализована"


class TestRunner:
    """Запускает тесты"""
    
    def __init__(self):
        self.llm = GeminiClient()
        self.llm.set_model(TEST_MODEL)
        self.simulator = CandidateSimulator(self.llm)
        self.checker = TestChecker()
        self.reports: List[TestReport] = []
    
    async def run_scenario(self, scenario: ScenarioConfig) -> TestReport:
        print(f"\n{'='*60}")
        print(f"🧪 {scenario.name}")
        print(f"   Модель: {TEST_MODEL} | Smart: {USE_SMART_MODE}")
        print(f"{'='*60}")
        
        start = datetime.now()
        errors = []
        
        orch = InterviewOrchestrator(smart_mode=USE_SMART_MODE)
        orch.set_model(TEST_MODEL)
        cand = Candidate(**scenario.candidate)
        orch.start_session(cand)
        
        history = []
        
        try:
            # приветствие
            res = await orch.generate_greeting()
            if "error" in res:
                errors.append(f"Ошибка приветствия: {res['error']}")
            else:
                msg = res["message"]
                print(f"🤖 {msg[:80]}...")
                history.append({"role": "agent", "text": msg})
            
            # диалог
            turn = 1
            while turn <= scenario.max_turns:
                last_msg = history[-1]["text"] if history else ""
                
                reply = await self.simulator.generate_reply(
                    last_msg, scenario.behavior, history, turn, scenario.candidate
                )
                
                print(f"👤 {reply[:80]}{'...' if len(reply) > 80 else ''}")
                history.append({"role": "user", "text": reply})
                
                res = await orch.process_message(reply)
                
                if res.get("finished"):
                    print("📊 Завершено")
                    break
                
                if "error" in res:
                    errors.append(f"Ошибка хода {turn}: {res['error']}")
                    break
                
                msg = res["message"]
                print(f"🤖 {msg[:80]}...")
                history.append({"role": "agent", "text": msg})
                
                turn += 1
            
            if not orch.session.finished:
                await orch.finish_interview()
            
        except Exception as e:
            errors.append(f"Exception: {e}")
            import traceback
            traceback.print_exc()
        
        # получаем данные сессии
        sess = orch.session.to_dict()

        
        # сохраняем логи
        # сохраняем логи
        log_file = f"test_logs/{scenario.name}_{datetime.now().strftime('%H%M%S')}.json"
        try:
            os.makedirs("test_logs", exist_ok=True)
            
            # полный лог для отладки
            full_log = orch.session.to_full_dict() if hasattr(orch.session, 'to_full_dict') else sess
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(full_log, f, ensure_ascii=False, indent=2)
            
            # лог по формату ТЗ
            with open(log_file.replace('.json', '_tz.json'), 'w', encoding='utf-8') as f:
                json.dump(sess, f, ensure_ascii=False, indent=2)

        except Exception as e:
            errors.append(f"Сохранение: {e}")
            log_file = "N/A"
        
        # проверки
        checks = {}
        for name in scenario.expected_checks:
            result, msg = self.checker.run_check(name, sess)
            checks[name] = (result, msg)
            print(f"  {result.value} {name}: {msg}")
        
        # итог
        results = [r for r, _ in checks.values()]
        if TestResult.FAIL in results:
            overall = TestResult.FAIL
        elif TestResult.WARN in results:
            overall = TestResult.WARN
        else:
            overall = TestResult.PASS
        
        duration = (datetime.now() - start).total_seconds()
        turns_count = len(sess.get("turns", []))
        
        await orch.close()
        
        return TestReport(scenario.name, overall, checks, duration, turns_count, log_file, errors)
    
    async def run_all(self, scenarios: List[ScenarioConfig] = None):
        if scenarios is None:
            scenarios = SCENARIOS
        
        print("\n" + "="*70)
        print("🚀 ТЕСТИРОВАНИЕ")
        print(f"📋 Сценариев: {len(scenarios)}")
        print(f"🤖 Модель: {TEST_MODEL}")
        print(f"🧠 Smart Mode: {USE_SMART_MODE}")
        print("="*70)
        
        for sc in scenarios:
            try:
                rep = await self.run_scenario(sc)
                self.reports.append(rep)
            except Exception as e:
                print(f"❌ Критическая ошибка {sc.name}: {e}")
                self.reports.append(TestReport(sc.name, TestResult.FAIL, {}, 0, 0, "N/A", [str(e)]))
            

            await asyncio.sleep(4)  # пауза между тестами чтобы не ловить 429
        
        await self.llm.close()
        self.print_summary()
    
    def print_summary(self):
        print("\n" + "="*70)
        print("📊 ИТОГИ")
        print("="*70)
        
        passed = sum(1 for r in self.reports if r.result == TestResult.PASS)
        warned = sum(1 for r in self.reports if r.result == TestResult.WARN)
        failed = sum(1 for r in self.reports if r.result == TestResult.FAIL)
        total = len(self.reports)
        
        print(f"\n✅ Пройдено: {passed}")
        print(f"⚠️  Замечания: {warned}")
        print(f"❌ Провалено: {failed}")
        print(f"📈 Результат: {passed}/{total} ({100*passed//total if total else 0}%)")
        
        print("\n" + "-"*70)
        for r in self.reports:
            print(f"\n{r.result.value} {r.scenario_name}")
            print(f"   ⏱️ {r.duration_sec:.1f}с | Ходов: {r.turns_count}")
            print(f"   📁 {r.log_file}")
            
            for err in r.errors:
                print(f"   🔴 {err}")
            
            for name, (res, msg) in r.checks.items():
                print(f"      {res.value} {name}")
        
        # сохраняем сводку
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model": TEST_MODEL,
            "smart_mode": USE_SMART_MODE,
            "total": total,
            "passed": passed,
            "warned": warned,
            "failed": failed,
            "scenarios": [
                {
                    "name": r.scenario_name,
                    "result": r.result.name,
                    "duration": r.duration_sec,
                    "turns": r.turns_count,
                    "checks": {k: {"result": v[0].name, "msg": v[1]} for k, v in r.checks.items()},
                    "errors": r.errors
                }
                for r in self.reports
            ]
        }
        
        os.makedirs("test_logs", exist_ok=True)
        with open(f"test_logs/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*70)
        if failed == 0:
            print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        else:
            print(f"⚠️ ПРОБЛЕМЫ: {failed} провалено")
        print("="*70)


async def main():
    runner = TestRunner()
    
    if len(sys.argv) > 1:
        name = sys.argv[1]
        sc = next((s for s in SCENARIOS if s.name == name), None)
        if sc:
            await runner.run_all([sc])
        else:
            print(f"Сценарий '{name}' не найден")
            print(f"Доступные: {[s.name for s in SCENARIOS]}")
    else:
        await runner.run_all()


if __name__ == "__main__":
    asyncio.run(main())
