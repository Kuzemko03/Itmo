from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Set, Any

@dataclass
class Candidate:
    name: str
    position: str
    grade: str
    experience: str

@dataclass
class Thought:
    agent: str
    text: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {"agent": self.agent, "thought": self.text}

@dataclass
class TurnData:
    turn_id: int
    user_message: str
    thoughts: List[Thought]
    agent_message: str
    difficulty: int
    flags: List[str]
    quality: str = ""
    
    def to_dict(self) -> Dict:
        thoughts_str = "\n".join([
            f"[{t.agent}]: {t.text.split(chr(10))[0][:550]}" 
            for t in self.thoughts
        ])
        return {
            "turn_id": self.turn_id,
            "agent_visible_message": self.agent_message,
            "user_message": self.user_message,
            "internal_thoughts": thoughts_str
        }


@dataclass
class SkillRecord:
    topic: str
    evidence: str
    turn_id: int
    score: int = 5
    
    def to_dict(self) -> Dict:
        return {"topic": self.topic, "evidence": self.evidence[:100], "score": self.score}

@dataclass
class GapRecord:
    topic: str
    question: str
    candidate_answer: str
    correct_answer: str
    turn_id: int
    severity: str = "medium"
    
    def to_dict(self) -> Dict:
        return {
            "topic": self.topic, "question_asked": self.question,
            "candidate_answer": self.candidate_answer[:100],
            "correct_answer": self.correct_answer, "severity": self.severity
        }

@dataclass
class FeedbackReport:
    decision: Dict[str, Any]
    technical: Dict[str, Any]
    soft_skills: Dict[str, Any]
    roadmap: Dict[str, Any]
    red_flags: List[str]
    green_flags: List[str]
    summary: str
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "generated_at": self.generated_at, "decision": self.decision,
            "technical_review": self.technical, "soft_skills_review": self.soft_skills,
            "roadmap": self.roadmap, "red_flags": self.red_flags,
            "green_flags": self.green_flags, "summary": self.summary
        }

@dataclass
class InterviewSession:
    candidate: Candidate
    started_at: str
    turns: List[TurnData] = field(default_factory=list)
    difficulty: int = 2
    topics_covered: Set[str] = field(default_factory=set)
    skills: List[SkillRecord] = field(default_factory=list)
    gaps: List[GapRecord] = field(default_factory=list)
    all_flags: List[str] = field(default_factory=list)
    feedback: Optional[FeedbackReport] = None
    finished: bool = False
    
    def to_dict(self) -> Dict:
        """Формат строго по ТЗ с ПОЛНЫМ feedback"""
        result = {
            "participant_name": self.candidate.name,
            "turns": [t.to_dict() for t in self.turns],
            "final_feedback": ""
        }
        
        if self.feedback:
            fb = self.feedback
            lines = []
            
            # Вердикт
            dec = fb.decision
            if dec:
                lines.append("ВЕРДИКТ")
                lines.append(f"Грейд: {dec.get('evaluated_grade', 'N/A')}")
                lines.append(f"Рекомендация: {dec.get('hiring_recommendation', 'N/A')}")
                lines.append(f"Уверенность: {dec.get('confidence_score', 'N/A')}%")
                lines.append(f"Обоснование: {dec.get('explanation', '')}")
                lines.append("")
            
            # Технические навыки
            tech = fb.technical
            if tech:
                lines.append(f"ТЕХНИЧЕСКИЕ НАВЫКИ ({tech.get('overall_score', 'N/A')}/10)")
                for skill in tech.get("confirmed_skills", [])[:8]:
                    if isinstance(skill, dict):
                        lines.append(f"✅ {skill.get('topic', '')} ({skill.get('score', 5)}/10)")
                    else:
                        lines.append(f"✅ {skill}")
                gaps = tech.get("knowledge_gaps", [])
                if gaps:
                    lines.append("Пробелы:")
                    for gap in gaps[:8]:
                        if isinstance(gap, dict):
                            lines.append(f"❌ {gap.get('topic', '')} [{gap.get('severity', 'medium')}]")
                        else:
                            lines.append(f"❌ {gap}")
                lines.append("")
            
            # Скиллы
            soft = fb.soft_skills
            if soft:
                lines.append("SOFT SKILLS")
                for name, data in soft.items():
                    if isinstance(data, dict) and "score" in data:
                        lines.append(f"{name}: {data['score']}/10")
                lines.append("")
            
            # Roadmap с ссылками
            road = fb.roadmap
            if road and road.get("priority_topics"):
                lines.append("ПЛАН РАЗВИТИЯ")
                for topic in road["priority_topics"][:5]:
                    if isinstance(topic, dict):
                        t_name = topic.get("topic", "")
                        t_res = topic.get("resources", [])
                        lines.append(f"📚 {t_name}")
                        for r in t_res[:2]:
                            lines.append(f"   🔗 {r}")
                lines.append("")
            
            # Флаги
            if fb.red_flags:
                lines.append("🔴 КРАСНЫЕ ФЛАГИ")
                for flag in fb.red_flags:
                    lines.append(f"   {flag}")
                lines.append("")
            
            if fb.green_flags:
                lines.append("🟢 ЗЕЛЁНЫЕ ФЛАГИ")
                for flag in fb.green_flags:
                    lines.append(f"   {flag}")
                lines.append("")
            
            if fb.summary:
                lines.append("РЕЗЮМЕ")
                lines.append(fb.summary)
            
            result["final_feedback"] = "\n".join(lines)
        
        return result
