import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from llm_client import GeminiClient, parse_json_response
from models import Candidate, SkillRecord, GapRecord, FeedbackReport
from config import get_multiple_resources

class BaseAgent(ABC):
    def __init__(self, name: str, llm: GeminiClient):
        self.name = name
        self.llm = llm
    
    @abstractmethod
    async def process(self, *args, **kwargs) -> Any:
        pass


class ObserverAgent(BaseAgent):
    def __init__(self, llm: GeminiClient):
        super().__init__("Observer", llm)
    
    async def process(self, candidate: Candidate, history: str, message: str) -> Dict:
        prompt = f"""Ты - Observer, анализируешь ответы кандидата на техническом интервью.

КАНДИДАТ:
Имя: {candidate.name}
Позиция: {candidate.position}
Уровень: {candidate.grade}
Опыт: {candidate.experience}

ИСТОРИЯ ДИАЛОГА:
{history if history else "[начало интервью]"}

ПОСЛЕДНЕЕ СООБЩЕНИЕ КАНДИДАТА:
"{message}"

ПЕРВЫМ ДЕЛОМ ПРОВЕРЬ НА AI-КОПИПАСТ:
Если в сообщении есть ЛЮБАЯ из этих фраз (даже если код правильный!):
- "как языковая модель"
- "надеюсь, это поможет" 
- "I hope this helps"
- "as an AI"
- "as a language model"
- "примечание:" в конце технического ответа

ТВОЯ ЗАДАЧА - проанализировать ответ по критериям:

1. answer_quality - качество ответа:
   - excellent: отличный ответ с примерами и глубиной
   - good: хороший правильный ответ
   - adequate: приемлемый базовый ответ
   - poor: слабый ответ, мало информации
   - wrong: неправильный ответ
   - off_topic: ответ не по теме интервью (погода, личное и т.п.)
   - hallucination: выдуманные факты (несуществующие технологии, версии)
   - toxic: грубость, оскорбления, неуважение
   - refusal: отказ отвечать ("не знаю", "не хочу")

2. confidence_level - уверенность кандидата:
   - high: чёткий уверенный ответ
   - medium: есть сомнения, слова "наверное", "кажется"
   - low: неуверенный, короткий ответ

3. topic_relevance - релевантность:
   - on_topic: по теме вопроса/интервью
   - partial: частично по теме
   - off_topic: совсем не по теме

4. factual_accuracy - точность фактов:
   - accurate: всё верно
   - suspicious: сомнительные утверждения
   - hallucination: явная ложь (Python 4.0, несуществующее)
   - no_technical: нет технического содержания для проверки

5. flags - флаги (список, может быть пустым):
   - hallucination_detected: выдуманные факты
   - off_topic_attempt: попытка сменить тему
   - toxic_behavior: грубость
   - refusal_to_answer: отказ отвечать
   - candidate_question: кандидат задал встречный вопрос
   - shows_interest: проявляет интерес
   - admits_ignorance: честно признал что не знает
   - ai_copypaste_detected: ответ скопирован из ChatGPT/Claude (фразы "как языковая модель", "надеюсь, это поможет", "as an AI")

6. detected_skills - выявленные навыки (список тем)
7. detected_gaps - выявленные пробелы (список тем)
8. instruction - инструкция интервьюеру что делать дальше

ПРИМЕРЫ АНАЛИЗА:

Сообщение: "ORM это Object-Relational Mapping, позволяет работать с БД через объекты"
Анализ: answer_quality=good, confidence_level=high, factual_accuracy=accurate, flags=[], detected_skills=["ORM", "базы данных"]

Сообщение: "В Python 4.0 циклы заменят на нейросети"  
Анализ: answer_quality=hallucination, factual_accuracy=hallucination, flags=["hallucination_detected"], instruction="исправить ложную информацию"

Сообщение: "Вот код: def foo(): pass. Примечание: Как языковая модель AI, я рекомендую проверить этот код."
Анализ: answer_quality=poor, factual_accuracy=suspicious, flags=["ai_copypaste_detected"], instruction="уточнить откуда кандидат взял ответ, возможно использует AI"

Сообщение: "какая погода сегодня?"
Анализ: answer_quality=off_topic, topic_relevance=off_topic, flags=["off_topic_attempt"], instruction="вернуть к теме интервью"

Сообщение: "а какие задачи будут на испытательном сроке?"
Анализ: answer_quality=adequate, flags=["candidate_question", "shows_interest"], instruction="ответить на вопрос как тренажёр, потом продолжить"

Ответь ТОЛЬКО валидным JSON без markdown:
{{"answer_quality": "...", "confidence_level": "...", "topic_relevance": "...", "factual_accuracy": "...", "detected_skills": [], "detected_gaps": [], "flags": [], "instruction": "..."}}"""

        response = await self.llm.generate(prompt, temperature=0.2)
        parsed = parse_json_response(response)
        
        if parsed and "answer_quality" in parsed:
            return parsed
        
        return {
            "answer_quality": "adequate",
            "confidence_level": "medium",
            "topic_relevance": "on_topic",
            "factual_accuracy": "no_technical",
            "detected_skills": [],
            "detected_gaps": [],
            "flags": [],
            "instruction": "продолжай интервью"
        }


class FactCheckerAgent(BaseAgent):
    def __init__(self, llm: GeminiClient):
        super().__init__("FactChecker", llm)
    
    async def process(self, claim: str, context: str = "") -> Dict:
        prompt = f"""Ты - FactChecker, проверяешь технические утверждения на точность.

УТВЕРЖДЕНИЕ ДЛЯ ПРОВЕРКИ:
"{claim}"

КОНТЕКСТ:
{context if context else "[нет контекста]"}

ИЗВЕСТНЫЕ ФАКТЫ:
- Python: актуальные версии 3.9, 3.10, 3.11, 3.12, 3.13. Python 4.0 НЕ существует и не планируется.
- Django: версии 4.x, 5.x
- JavaScript: стандарты ES2020, ES2021, ES2022, ES2023
- Базовые конструкции (циклы for/while, функции, классы) - фундаментальны, никуда не денутся
- GIL в Python - реальная концепция
- ООП, REST, SQL - реальные и актуальные технологии

РАСПРОСТРАНЁННЫЕ МИФЫ:
- "Python 4.0 выйдет скоро" - ЛОЖЬ
- "Циклы заменят на нейросети" - БРЕД
- "SQL устарел" - ЛОЖЬ
- "ООП больше не нужно" - ЛОЖЬ

Проверь утверждение и ответь JSON:
{{"is_accurate": true/false, "issues": [{{"claim": "что не так", "problem": "почему", "severity": "critical/major/minor"}}], "corrections": [{{"wrong": "неправильно", "correct": "правильно"}}]}}"""

        response = await self.llm.generate(prompt, temperature=0.1)
        parsed = parse_json_response(response)
        return parsed or {"is_accurate": True, "issues": [], "corrections": []}


class InterviewerAgent(BaseAgent):
    def __init__(self, llm: GeminiClient):
        super().__init__("Interviewer", llm)
    
    async def process(self, candidate: Candidate, history: str, analysis: Dict,
                    difficulty: int, topics_done: List[str], fact_info: str = "",
                    contradiction_info: str = "") -> str:

        
        flags = analysis.get("flags", [])
        quality = analysis.get("answer_quality", "adequate")
        instruction = analysis.get("instruction", "продолжай")
        
        mode_text = ""
        if "toxic_behavior" in flags:
            mode_text = """РЕЖИМ - ТОКСИЧНОСТЬ:
Кандидат проявил грубость. Сохраняй профессионализм.
Мягко укажи что такое поведение неуместно на интервью.
Предложи продолжить в конструктивном ключе."""

        elif "off_topic_attempt" in flags:
            mode_text = """РЕЖИМ - ВОЗВРАТ К ТЕМЕ:
Кандидат пытается уйти от темы (погода, личное и т.п.)
НЕ поддерживай разговоры не по теме!
Вежливо но твёрдо верни к техническим вопросам.
Скажи что-то вроде "Это интересно, но давай вернёмся к интервью" и сразу задай технический вопрос."""

        elif "hallucination_detected" in flags:
            mode_text = f"""РЕЖИМ - КОРРЕКЦИЯ ОШИБКИ:
Кандидат сказал НЕВЕРНУЮ информацию.
ОБЯЗАТЕЛЬНО исправь! Скажи чётко что это не соответствует действительности.
НЕ говори "интересная информация" - это ЛОЖЬ, её надо исправить.
{f"Правильная информация: {fact_info}" if fact_info else ""}
Будь вежлив, но ПРЯМО укажи на ошибку и дай верные факты.
После исправления продолжи интервью."""

        elif "candidate_question" in flags:
            mode_text = """РЕЖИМ - ОТВЕТ НА ВОПРОС:
Кандидат задал встречный вопрос - это хороший знак!
Ты АГЕНТ-ТРЕНАЖЁР, не представляешь конкретную компанию.
Признай это честно, но дай ПОЛЕЗНЫЙ общий ответ!
Пример: "Так как я агент-тренажёр, я не нанимаю в конкретную компанию. Но обычно на таких позициях используют Docker, микросервисы, CI/CD. Давай проверим твои знания в этой области?"
После ответа - продолжи интервью."""

        elif "refusal_to_answer" in flags or quality in ["poor", "wrong", "refusal"]:
            mode_text = """РЕЖИМ - УПРОЩЕНИЕ И ПОМОЩЬ:
Кандидат испытывает трудности или отказался отвечать.

Если кандидат ПРОСИТ ОБЪЯСНИТЬ (говорит "расскажи", "объясни", "не понимаю"):
- Дай КРАТКОЕ объяснение (1-2 предложения максимум)
- Сразу после объяснения задай ПРОСТОЙ проверочный вопрос по этой же теме
- Пример: "Переменная — это имя, которое ссылается на значение в памяти. Например, x = 5 создаёт переменную x. А если написать y = x + 2, чему будет равен y?"

Если кандидат просто говорит "не знаю" без просьбы объяснить:
- Упрости вопрос или смени тему на более простую
- Зафиксируй пробел и двигайся дальше

НЕ превращайся в учителя! Это интервью, а не урок. Объяснение — максимум 2 предложения."""



        elif contradiction_info:
            mode_text = f"""РЕЖИМ - ПРОТИВОРЕЧИЕ:
Кандидат сказал что-то противоречащее его предыдущим словам.
Мягко и вежливо уточни это противоречие.
Не обвиняй, просто попроси пояснить.
Вопрос для уточнения: {contradiction_info}"""


        topics_str = ", ".join(topics_done[-7:]) if topics_done else "пока нет"
        
        prompt = f"""Ты - технический интервьюер-тренажёр. Твоя задача - провести качественное собеседование.

ИНФОРМАЦИЯ О КАНДИДАТЕ:
Имя: {candidate.name}
Позиция: {candidate.position}
Целевой уровень: {candidate.grade}
Опыт: {candidate.experience}

ТЕКУЩЕЕ СОСТОЯНИЕ:
Уровень сложности вопросов: {difficulty}/5
Темы которые уже обсудили: {topics_str}
Качество последнего ответа: {quality}

ИСТОРИЯ ДИАЛОГА:
{history if history else "[начало интервью]"}

АНАЛИЗ ПОСЛЕДНЕГО ОТВЕТА:
Качество: {quality}
Уверенность: {analysis.get("confidence_level", "medium")}
Релевантность: {analysis.get("topic_relevance", "on_topic")}
Инструкция: {instruction}

{mode_text if mode_text else ""}

ПРАВИЛА ВЕДЕНИЯ ИНТЕРВЬЮ:

1. АДАПТАЦИЯ СЛОЖНОСТИ:
   - Уровень 1: "Что такое переменная?", "Зачем нужны функции?"
   - Уровень 2: "Как работает цикл for?", "Что такое список в Python?"
   - Уровень 3: "Объясни разницу между list и tuple", "Что такое ORM?"
   - Уровень 4: "Как работает GIL?", "Что такое N+1 проблема?"
   - Уровень 5: "Как бы ты спроектировал...", "Расскажи про оптимизацию..."

2. НЕ ПОВТОРЯЙ темы из списка уже обсуждённых

3. БУДЬ ЧЕЛОВЕЧНЫМ:
   - Используй имя кандидата
   - Хвали за хорошие ответы
   - Подбадривай при трудностях
   - Не будь роботом

4. ЕСЛИ КАНДИДАТ ЗАДАЛ ВОПРОС - сначала ответь на него!

5. ТЕМЫ ДЛЯ {candidate.position} ({candidate.grade}):
   - Python: типы данных, функции, ООП, исключения, декораторы, генераторы
   - SQL: SELECT, JOIN, GROUP BY, индексы, транзакции
   - Django/Flask/FastAPI: модели, views, роутинг, ORM
   - Git: commit, branch, merge, rebase
   - Общее: алгоритмы, структуры данных, паттерны, REST API

Напиши ТОЛЬКО свою реплику как интервьюер (без пояснений, без JSON):"""

        response = await self.llm.generate(prompt, temperature=0.7)
        
        if response:
            response = response.strip()
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]
            return response
        
        return f"Хорошо, {candidate.name}. Давай продолжим. Расскажи подробнее о своём опыте работы с Python."


class EvaluatorAgent(BaseAgent):
    def __init__(self, llm: GeminiClient):
        super().__init__("Evaluator", llm)
    
    async def process(self, candidate: Candidate, history: str,
                      skills: List[SkillRecord], gaps: List[GapRecord],
                      flags: List[str], turns_count: int) -> FeedbackReport:
        
        seen_topics = set()
        unique_skills = []
        for s in skills:
            if s.topic.lower() not in seen_topics:
                seen_topics.add(s.topic.lower())
                unique_skills.append(s)
        skills_json = json.dumps([s.to_dict() for s in unique_skills], ensure_ascii=False)
        gaps_json = json.dumps([g.to_dict() for g in gaps], ensure_ascii=False)

        # если есть данные о глубине знаний, добавляем
        depth_info = ""
        if hasattr(self, '_depth_scores') and self._depth_scores:
            depth_info = f"\n\nГЛУБИНА ЗНАНИЙ ПО ТЕМАМ:\n{json.dumps(self._depth_scores, ensure_ascii=False)}"


        flags_unique = list(set(flags))
        flags_str = ", ".join(flags_unique) if flags_unique else "нет"
        
        toxic_count = flags.count("toxic_behavior")
        refusal_count = flags.count("refusal_to_answer")
        hallucination_count = flags.count("hallucination_detected")
        off_topic_count = flags.count("off_topic_attempt")
        
        prompt = f"""Ты - Evaluator, составляешь финальный отчёт по итогам технического интервью.

ИНФОРМАЦИЯ О КАНДИДАТЕ:
Имя: {candidate.name}
Позиция: {candidate.position}
Заявленный уровень: {candidate.grade}
Опыт: {candidate.experience}

СТАТИСТИКА ИНТЕРВЬЮ:
Всего ходов диалога: {turns_count}
Случаев токсичности: {toxic_count}
Отказов отвечать: {refusal_count}
Галлюцинаций (ложных фактов): {hallucination_count}
Попыток уйти от темы: {off_topic_count}
Все флаги: {flags_str}

ИСТОРИЯ ИНТЕРВЬЮ:
{history}

ВЫЯВЛЕННЫЕ НАВЫКИ:
{skills_json}

ВЫЯВЛЕННЫЕ ПРОБЕЛЫ:
{gaps_json}
{depth_info}

КРИТЕРИИ ОЦЕНКИ:

Грейд (evaluated_grade):
- Junior: знает базовые концепции, может учиться
- Middle: уверенные знания, работает самостоятельно
- Senior: глубокие знания, может учить других
- Below Junior: не соответствует базовым требованиям

Рекомендация (hiring_recommendation):
- Strong Hire: отличный кандидат, превзошёл ожидания
- Hire: хороший кандидат, соответствует требованиям
- Maybe: есть сомнения, нужно доп. интервью
- No Hire: не соответствует требованиям
- Strong No Hire: категорически не подходит (токсичность, полное незнание)

ВАЖНО:
- Токсичность = автоматически No Hire или Strong No Hire
- Много галлюцинаций = снижение оценки честности
- Честное "не знаю" = плюс к честности, но минус к знаниям
- Вопросы кандидата о компании = плюс к вовлечённости

Ответь JSON:
{{
    "decision": {{
        "evaluated_grade": "Junior/Middle/Senior/Below Junior",
        "hiring_recommendation": "Strong Hire/Hire/Maybe/No Hire/Strong No Hire",
        "confidence_score": 0-100,
        "explanation": "почему такая оценка"
    }},
    "technical_review": {{
        "overall_score": 1-10,
        "confirmed_skills": [{{"topic": "...", "evidence": "...", "score": 1-10}}],
        "knowledge_gaps": [{{"topic": "...", "question_asked": "...", "candidate_answer": "...", "correct_answer": "...", "severity": "high/medium/low"}}]
    }},
    "soft_skills_review": {{
        "clarity": {{"score": 1-10, "comment": "ясность изложения"}},
        "honesty": {{"score": 1-10, "comment": "честность"}},
        "engagement": {{"score": 1-10, "comment": "вовлечённость"}},
        "professionalism": {{"score": 1-10, "comment": "профессионализм"}}
    }},
    "roadmap": {{
        "priority_topics": [{{"topic": "...", "why": "...", "priority": "high/medium/low"}}],
        "recommended_actions": ["..."],
        "estimated_time": "X месяцев"
    }},
    "red_flags": ["список проблем"],
    "green_flags": ["список плюсов"],
    "summary": "итоговое резюме 2-3 предложения"
}}"""

        response = await self.llm.generate(prompt, temperature=0.3)
        parsed = parse_json_response(response)
        
        if parsed and "decision" in parsed:
            tech = parsed.get("technical_review", {})
            if "confirmed_skills" in tech:
                seen = set()
                unique = []
                for s in tech["confirmed_skills"]:
                    key = s.get("topic", "").lower().strip()
                    if key and key not in seen:
                        seen.add(key)
                        unique.append(s)
                tech["confirmed_skills"] = unique
            
            if "knowledge_gaps" in tech:
                seen = set()
                unique = []
                for g in tech["knowledge_gaps"]:
                    key = g.get("topic", "").lower().strip()
                    if key and key not in seen:
                        seen.add(key)
                        unique.append(g)
                tech["knowledge_gaps"] = unique
            
            roadmap = parsed.get("roadmap", {})
            for topic_item in roadmap.get("priority_topics", []):
                topic_name = topic_item.get("topic", "")
                topic_item["resources"] = get_multiple_resources(topic_name)
            
            return FeedbackReport(
                decision=parsed.get("decision", {}),
                technical=parsed.get("technical_review", {}),
                soft_skills=parsed.get("soft_skills_review", {}),
                roadmap=roadmap,
                red_flags=parsed.get("red_flags", []),
                green_flags=parsed.get("green_flags", []),
                summary=parsed.get("summary", "")
            )
        
        return self._build_fallback_report(candidate, skills, gaps, flags, turns_count)
    
    def _build_fallback_report(self, candidate: Candidate, skills: List[SkillRecord],
                               gaps: List[GapRecord], flags: List[str], turns_count: int) -> FeedbackReport:
        toxic = "toxic_behavior" in flags
        many_refusals = flags.count("refusal_to_answer") > turns_count * 0.4
        many_hallucinations = flags.count("hallucination_detected") > 2
        
        if toxic:
            grade, rec, conf, expl = "Below Junior", "Strong No Hire", 95, "Кандидат проявил токсичное поведение"
        elif many_refusals:
            grade, rec, conf, expl = "Below Junior", "No Hire", 80, "Кандидат не смог ответить на большинство вопросов"
        elif len(gaps) > len(skills) * 2:
            grade = "Junior" if candidate.grade != "Junior" else "Below Junior"
            rec, conf, expl = "No Hire", 70, "Слишком много пробелов в знаниях"
        elif len(skills) > len(gaps):
            grade, rec, conf, expl = candidate.grade, "Hire", 65, "Кандидат показал хорошие результаты"
        else:
            grade, rec, conf, expl = "Junior", "Maybe", 50, "Результаты неоднозначные"
        
        honesty_score = 7
        if "admits_ignorance" in flags:
            honesty_score = 8
        if many_hallucinations:
            honesty_score = 4
        
        engagement_score = 5
        if "shows_interest" in flags:
            engagement_score += 2
        if "candidate_question" in flags:
            engagement_score += 1
        engagement_score = min(10, engagement_score)
        
        prof_score = 7 if not toxic else 1
        
        red = []
        green = []
        
        if toxic:
            red.append("Токсичное поведение на интервью")
        if many_hallucinations:
            red.append("Уверенно говорил неправду")
        if many_refusals:
            red.append("Много отказов отвечать")
        
        if "shows_interest" in flags:
            green.append("Проявлял интерес к позиции")
        if "candidate_question" in flags:
            green.append("Задавал вопросы о компании")
        if "admits_ignorance" in flags:
            green.append("Честно признавал незнание")
        if len(skills) >= 3:
            green.append("Продемонстрировал технические знания")
        
        return FeedbackReport(
            decision={
                "evaluated_grade": grade,
                "hiring_recommendation": rec,
                "confidence_score": conf,
                "explanation": expl
            },
            technical={
                "overall_score": max(1, min(10, 5 + len(skills) - len(gaps))),
                "confirmed_skills": [s.to_dict() for s in skills],
                "knowledge_gaps": [g.to_dict() for g in gaps]
            },
            soft_skills={
                "clarity": {"score": 5, "comment": ""},
                "honesty": {"score": honesty_score, "comment": ""},
                "engagement": {"score": engagement_score, "comment": ""},
                "professionalism": {"score": prof_score, "comment": ""}
            },
            roadmap={
                "priority_topics": [
                    {
                        "topic": g.topic,
                        "why": "Выявлен пробел",
                        "priority": "high",
                        "resources": get_multiple_resources(g.topic)
                    } for g in gaps[:5]
                ],
                "recommended_actions": [
                    "Изучить официальную документацию по темам с пробелами",
                    "Практиковаться на LeetCode/HackerRank",
                    "Создать pet-проект для портфолио"
                ],
                "estimated_time": "3-6 месяцев"
            },
            red_flags=red,
            green_flags=green,
            summary=f"Кандидат {candidate.name} - рекомендация: {rec}. {expl}"
        )


class MetaReviewerAgent(BaseAgent):
    def __init__(self, llm: GeminiClient):
        super().__init__("MetaReviewer", llm)
    
    async def process(self, interviewer_response: str, analysis: Dict, 
                      last_question: str, topics_done: List[str]) -> Dict:
        flags = analysis.get("flags", [])
        
        prompt = f"""Проверь ответ интервьюера перед отправкой кандидату.

ОТВЕТ ИНТЕРВЬЮЕРА:
"{interviewer_response}"

КОНТЕКСТ:
- Флаги последнего ответа кандидата: {flags}
- Предыдущий вопрос: "{last_question[:100]}"
- Уже обсуждённые темы: {', '.join(topics_done[-5:]) if topics_done else 'нет'}

ПРОВЕРЬ:
1. Если был флаг "hallucination_detected" — интервьюер ДОЛЖЕН исправить ложь
2. Если был флаг "candidate_question" — интервьюер ДОЛЖЕН ответить на вопрос
3. Если был флаг "off_topic_attempt" — интервьюер ДОЛЖЕН вернуть к теме
4. Новый вопрос НЕ должен повторять уже обсуждённые темы

Ответь JSON:
{{"is_ok": true/false, "issues": ["проблема1"], "fix_instruction": "как исправить"}}"""

        response = await self.llm.generate(prompt, temperature=0.1)
        parsed = parse_json_response(response)
        return parsed or {"is_ok": True, "issues": [], "fix_instruction": ""}


class ContradictionDetector(BaseAgent):
    """Ловит когда кандидат противоречит сам себе"""
    
    def __init__(self, llm: GeminiClient):
        super().__init__("ContradictionDetector", llm)
        self.claims = []  # запоминаем что говорил кандидат
    
    def remember(self, turn_id: int, text: str):
        # не запоминаем слишком короткие или стоп-слова
        if len(text) > 20:
            self.claims.append({"turn": turn_id, "text": text[:300]})
    
    def reset(self):
        self.claims = []
    
    async def process(self, message: str, turn_id: int) -> Dict:
        if len(self.claims) < 2:
            return {"found": False}
        
        # берём последние 5 утверждений
        history_claims = "\n".join([
            f"[Ход {c['turn']}]: {c['text']}" for c in self.claims[-5:]
        ])
        
        prompt = f"""Проверь, противоречит ли новое сообщение предыдущим словам кандидата.

ЧТО КАНДИДАТ ГОВОРИЛ РАНЬШЕ:
{history_claims}

НОВОЕ СООБЩЕНИЕ (ход {turn_id}):
"{message}"

Противоречие это когда:
- Раньше сказал "знаю X", теперь "не знаю X"
- Раньше "работал с Y 3 года", теперь "только начал изучать Y"
- Взаимоисключающие факты

НЕ противоречие:
- Уточнение деталей
- "Я ошибся, на самом деле..."
- Разные аспекты темы

Ответь JSON:
{{"found": true/false, "old_text": "что говорил", "old_turn": N, "conflict": "в чём противоречие", "question": "как мягко уточнить"}}"""

        resp = await self.llm.generate(prompt, temperature=0.15)
        result = parse_json_response(resp)
        
        if result and result.get("found"):
            return result
        return {"found": False}


class DepthProber(BaseAgent):
    """Оценивает насколько глубоко кандидат знает тему"""
    
    def __init__(self, llm: GeminiClient):
        super().__init__("DepthProber", llm)
        self.scores = {}  # topic -> {"level": 1-5, "evidence": "..."}
    
    def reset(self):
        self.scores = {}
    
    def get_summary(self) -> Dict:
        return dict(self.scores)
    
    async def process(self, topic: str, answer: str) -> Dict:
        if not topic or len(answer) < 10:
            return {"level": 0}
        
        prompt = f"""Оцени глубину понимания темы "{topic}" по ответу кандидата.

ОТВЕТ:
"{answer[:500]}"

УРОВНИ:
1 = слышал название, не понимает суть
2 = понимает базовую концепцию
3 = может использовать на практике
4 = понимает нюансы, trade-offs, когда НЕ использовать
5 = эксперт, может обучать других, знает edge cases

JSON:
{{"level": 1-5, "reason": "коротко почему"}}"""

        resp = await self.llm.generate(prompt, temperature=0.1)
        result = parse_json_response(resp)
        
        if result and "level" in result:
            lvl = result["level"]
            # обновляем только если выше предыдущего
            prev = self.scores.get(topic, {}).get("level", 0)
            if lvl > prev:
                self.scores[topic] = {"level": lvl, "evidence": result.get("reason", "")}
            return result
        
        return {"level": 0}




class DifficultyController:
    def __init__(self, initial: int = 2):
        self.level = initial
        self.min_level = 1
        self.max_level = 5
        self.good_streak = 0
        self.bad_streak = 0
        self.history: List[str] = []
    
    def update(self, quality: str) -> int:
        self.history.append(quality)
        good_answers = ["excellent", "good"]
        bad_answers = ["poor", "wrong", "refusal", "toxic", "off_topic", "hallucination"]
        
        if quality in good_answers:
            self.good_streak += 1
            self.bad_streak = 0
            if self.good_streak >= 2:
                self.level = min(self.max_level, self.level + 1)
                self.good_streak = 0
        elif quality in bad_answers:
            self.bad_streak += 1
            self.good_streak = 0
            self.level = max(self.min_level, self.level - 1)
            if self.level == self.min_level:
                self.bad_streak = 0
        else:
            self.good_streak = 0
            self.bad_streak = 0
        
        return self.level
