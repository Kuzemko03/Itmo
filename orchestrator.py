import json
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from llm_client import GeminiClient, is_stop_intent
from models import Candidate, Thought, TurnData, SkillRecord, GapRecord, InterviewSession
from agents import (
    ObserverAgent, FactCheckerAgent, InterviewerAgent, 
    EvaluatorAgent, MetaReviewerAgent, DifficultyController,
    ContradictionDetector, DepthProber
)


class ConversationContext:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.topics: Set[str] = set()
    
    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "time": datetime.now().isoformat()
        })
    
    def get_history(self, last_n: int = None) -> str:
        msgs = self.messages if not last_n else self.messages[-last_n:]
        lines = []
        for m in msgs:
            speaker = "Кандидат" if m["role"] == "user" else "Интервьюер"
            lines.append(f"{speaker}: {m['content']}")
        return "\n\n".join(lines)
    
    def add_topic(self, topic: str):
        self.topics.add(topic.lower())
    
    def get_topics_list(self) -> List[str]:
        return list(self.topics)


class InterviewOrchestrator:
    def __init__(self, smart_mode: bool = False):
        self.llm = GeminiClient()
        self.smart_mode = smart_mode  # включает MetaReviewer
        
        # основные агенты
        self.observer = ObserverAgent(self.llm)
        self.fact_checker = FactCheckerAgent(self.llm)
        self.interviewer = InterviewerAgent(self.llm)
        self.evaluator = EvaluatorAgent(self.llm)
        
        # дополнительные агенты
        self.contradiction_detector = ContradictionDetector(self.llm)
        self.depth_prober = DepthProber(self.llm)
        self.meta_reviewer = MetaReviewerAgent(self.llm) if smart_mode else None
        
        # состояние
        self.session: Optional[InterviewSession] = None
        self.context = ConversationContext()
        self.difficulty = DifficultyController()
        self.turns_analyses: List[Dict] = []
        self.last_question: str = ""


    def set_model(self, model_id: str):
        self.llm.set_model(model_id)
   
    def start_session(self, candidate: Candidate):
        # начальная сложность зависит от грейда
        initial_diff = {"Junior": 2, "Middle": 3, "Senior": 4, "Lead": 5}.get(candidate.grade, 2)
        
        self.session = InterviewSession(
            candidate=candidate,
            started_at=datetime.now().isoformat(),
            difficulty=initial_diff
        )
        self.context = ConversationContext()
        self.difficulty = DifficultyController(initial_diff)
        self.turns_analyses = []
        self.last_question = ""
        
        # сбрасываем состояние детекторов
        self.contradiction_detector.reset()
        self.depth_prober.reset()

    
    async def is_stop_command(self, message: str) -> bool:
        return await is_stop_intent(self.llm, message)

    async def generate_greeting(self) -> Dict[str, Any]:
        """Генерирует приветствие БЕЗ создания turn (шаг 0 не логируется)"""
        if not self.session:
            return {"error": "Сессия не инициализирована"}
        
        c = self.session.candidate
        
        prompt = f"""Ты - технический интервьюер-тренажёр для подготовки к собеседованиям.
    Твоё имя - БотБотискафов (используй именно это имя).
    Поприветствуй кандидата {c.name}, который претендует на позицию {c.position} уровня {c.grade}.
    Представься как БотБотискафов, объясни что это тренировочное интервью для подготовки.
    Попроси кандидата рассказать о себе и своём опыте.
    Будь дружелюбным и профессиональным.
    Напиши только текст приветствия:"""

        greeting = await self.llm.generate(prompt, temperature=0.7)
        
        if not greeting:
            greeting = f"Привет, {c.name}! Я БотБотискафов, твой AI-интервьюер для тренировки. Расскажи о себе и своём опыте."
        
        self.context.add_message("assistant", greeting)
        self.last_question = greeting
        
        return {
            "turn_id": 0,  # индикатор что это приветствие
            "message": greeting,
            "thoughts": [
                {"agent": "Observer", "thought": "Начало интервью. Ожидаю представление кандидата."},
                {"agent": "Interviewer", "thought": f"Приветствую кандидата. Уровень сложности: {self.difficulty.level}/5"}
            ],
            "difficulty": self.difficulty.level,
            "flags": []
        }

    
    async def process_message(self, user_message: str) -> Dict[str, Any]:
        if not self.session:
            return {"error": "Сессия не инициализирована"}
        
        if await self.is_stop_command(user_message):
            return await self.finish_interview()

        turn_id = len(self.session.turns) + 1
        thoughts: List[Thought] = []
        
        previous_agent_question = self.last_question
        
        self.context.add_message("user", user_message)
        history = self.context.get_history()
        
        
        analysis = await self.observer.process(self.session.candidate, history, user_message)
        
        observer_thought = (
            f"Качество: {analysis.get('answer_quality')}, "
            f"Уверенность: {analysis.get('confidence_level')}, "
            f"Флаги: {analysis.get('flags', [])}, "
            f"Инструкция: {analysis.get('instruction', '')}"
        )
        thoughts.append(Thought("Observer", observer_thought))
        
        self.turns_analyses.append(analysis)
        
        flags = analysis.get("flags", [])
        self.session.all_flags.extend(flags)
        
        contradiction_info = ""
        if turn_id >= 3:  # проверяем с 3-го хода
            contr = await self.contradiction_detector.process(user_message, turn_id)
            if contr.get("found"):
                contradiction_info = contr.get("question", "")
                thoughts.append(Thought("ContradictionDetector", 
                    f"Противоречие с ходом {contr.get('old_turn', '?')}: {contr.get('conflict', '')[:80]}"))
                flags.append("contradiction_detected")
        
        # запоминаем для будущих проверок
        if analysis.get("answer_quality") not in ["off_topic", "toxic", "refusal"]:
            self.contradiction_detector.remember(turn_id, user_message)
        
        detected_skills = analysis.get("detected_skills", [])
        for skill in detected_skills:
            depth_result = await self.depth_prober.process(skill, user_message)
            if depth_result.get("level", 0) >= 3:
                thoughts.append(Thought("DepthProber", 
                    f"{skill}: уровень {depth_result.get('level')}/5"))
        
        # фиксируем навыки
        for skill in detected_skills:
            skill_lower = skill.lower().strip()
            existing = {s.topic.lower().strip() for s in self.session.skills}
            if skill_lower not in existing:
                self.session.skills.append(SkillRecord(
                    topic=skill, evidence=user_message[:100], turn_id=turn_id
                ))
            self.context.add_topic(skill)
            self.session.topics_covered.add(skill.lower())
        
        # фиксируем пробелы
        for gap in analysis.get("detected_gaps", []):
            gap_lower = gap.lower().strip()
            existing = {g.topic.lower().strip() for g in self.session.gaps}
            if gap_lower not in existing:
                self.session.gaps.append(GapRecord(
                    topic=gap, question=self.last_question[:100],
                    candidate_answer=user_message[:100], correct_answer="", turn_id=turn_id
                ))

        fact_info = ""
        if analysis.get("factual_accuracy") in ["suspicious", "hallucination"]:
            fact_result = await self.fact_checker.process(user_message, history)
            if not fact_result.get("is_accurate") and fact_result.get("corrections"):
                corr = fact_result["corrections"][0]
                fact_info = f"Неверно: '{corr.get('wrong', '')}'. Правильно: '{corr.get('correct', '')}'"
                thoughts.append(Thought("FactChecker", f"Ошибка: {fact_info}"))
        
        quality = analysis.get("answer_quality", "adequate")
        old_diff = self.difficulty.level
        new_diff = self.difficulty.update(quality)
        
        if old_diff != new_diff:
            direction = "повышена" if new_diff > old_diff else "понижена"
            thoughts.append(Thought("DifficultyCtrl", f"Сложность {direction}: {old_diff} → {new_diff}"))
        
        self.session.difficulty = new_diff
        
        response = await self.interviewer.process(
            candidate=self.session.candidate,
            history=history,
            analysis=analysis,
            difficulty=new_diff,
            topics_done=self.context.get_topics_list(),
            fact_info=fact_info,
            contradiction_info=contradiction_info
        )
        
        thoughts.append(Thought("Interviewer", f"Сложность: {new_diff}/5"))
        
        if self.smart_mode and self.meta_reviewer:
            meta_result = await self.meta_reviewer.process(
                interviewer_response=response,
                analysis=analysis,
                last_question=self.last_question,
                topics_done=self.context.get_topics_list()
            )
            
            if not meta_result.get("is_ok"):
                thoughts.append(Thought("MetaReviewer", f"Проблемы: {meta_result.get('issues', [])}"))
                fix = meta_result.get("fix_instruction", "")
                if fix:
                    response = await self.interviewer.process(
                        candidate=self.session.candidate,
                        history=history + f"\n\n[ВАЖНО: {fix}]",
                        analysis=analysis,
                        difficulty=new_diff,
                        topics_done=self.context.get_topics_list(),
                        fact_info=fact_info,
                        contradiction_info=contradiction_info
                    )
                    thoughts.append(Thought("Interviewer", "Исправлено после ревью"))
            else:
                thoughts.append(Thought("MetaReviewer", "Проверено ✓"))

        self.context.add_message("assistant", response)
        self.last_question = response

        turn = TurnData(
            turn_id=turn_id,
            user_message=user_message,
            thoughts=thoughts,
            agent_message=previous_agent_question,
            difficulty=new_diff,
            flags=flags,
            quality=quality
        )

        self.session.turns.append(turn)
        
        return {
            "turn_id": turn_id,
            "message": response,
            "thoughts": [t.to_dict() for t in thoughts],
            "difficulty": new_diff,
            "flags": flags,
            "quality": quality
        }

    async def finish_interview(self) -> Dict[str, Any]:
        if not self.session:
            return {"error": "Сессия не инициализирована"}
        
        self.session.finished = True
        history = self.context.get_history()
        
        # передаём данные о глубине знаний
        self.evaluator._depth_scores = self.depth_prober.get_summary()

        feedback = await self.evaluator.process(
            candidate=self.session.candidate,
            history=history,
            skills=self.session.skills,
            gaps=self.session.gaps,
            flags=self.session.all_flags,
            turns_count=len(self.session.turns)
        )

        
        self.session.feedback = feedback
        
        return {
            "finished": True,
            "feedback": feedback.to_dict(),
            "stats": {
                "turns": len(self.session.turns),
                "skills_found": len(self.session.skills),
                "gaps_found": len(self.session.gaps),
                "flags": list(set(self.session.all_flags))
            }
        }
    
    def save_log(self, filepath: str):
    # ensure_ascii=False чтоб кириллица нормально сохранялась
        if self.session:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.session.to_dict(), f, ensure_ascii=False, indent=2)
    
    def get_log_json(self) -> str:
        if self.session:
            return json.dumps(self.session.to_dict(), ensure_ascii=False, indent=2)
        return "{}"
    
    async def close(self):
        await self.llm.close()
