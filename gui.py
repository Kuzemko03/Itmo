import json
import asyncio
import threading # для async в tkinter
from datetime import datetime
from typing import Dict, List

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog
    HAS_GUI = True
except ImportError:
    HAS_GUI = False

from config import AVAILABLE_MODELS, adapt_log_to_tz_format
from models import Candidate
from orchestrator import InterviewOrchestrator

class InterviewGUI:
    def __init__(self):
        self.orchestrator = None
        self.loop = asyncio.new_event_loop()
        self.root = tk.Tk()
        self.root.title("Interview Coach")
        self.root.geometry("1100x750")
        self._setup_ui()
        self._disable_chat()
    
    def _setup_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        
        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(top, text="😋 Новое", command=self._new_interview).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="🤓 Стоп", command=self._stop_interview).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="✅ Сохранить", command=self._save_log).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(top, text="Модель:").pack(side=tk.LEFT, padx=(20, 5))
        self.model_var = tk.StringVar(value="Gemini 3 Flash (рекомендуется)")
        ttk.Combobox(top, textvariable=self.model_var, values=list(AVAILABLE_MODELS.keys()), state="readonly", width=25).pack(side=tk.LEFT)
        
        self.smart_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="🧠 Smart", variable=self.smart_var).pack(side=tk.LEFT, padx=15)
        
        self.status_var = tk.StringVar(value="Нажмите 'Новое'")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.RIGHT, padx=10)
        self.diff_var = tk.StringVar(value="Сложность: -")
        ttk.Label(top, textvariable=self.diff_var).pack(side=tk.RIGHT, padx=10)
        
        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        left = ttk.LabelFrame(paned, text="Диалог", padding=5)
        paned.add(left, weight=1)
        self.chat_area = scrolledtext.ScrolledText(left, wrap=tk.WORD, font=("Arial", 11), state=tk.DISABLED)
        self.chat_area.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.chat_area.tag_configure("user", foreground="#0066cc")
        self.chat_area.tag_configure("agent", foreground="#009933")
        self.chat_area.tag_configure("system", foreground="#666666")
        
        inp = ttk.Frame(left)
        inp.pack(fill=tk.X)
        self.input_entry = ttk.Entry(inp, font=("Arial", 11))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_entry.bind("<Return>", lambda e: self._send())
        self.send_btn = ttk.Button(inp, text="Отправить", command=self._send)
        self.send_btn.pack(side=tk.RIGHT)
        
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        nb = ttk.Notebook(right)
        nb.pack(fill=tk.BOTH, expand=True)
        
        tf = ttk.Frame(nb, padding=5)
        nb.add(tf, text="🧠 Мысли")
        self.thoughts_area = scrolledtext.ScrolledText(tf, wrap=tk.WORD, font=("Consolas", 10), state=tk.DISABLED)
        self.thoughts_area.pack(fill=tk.BOTH, expand=True)
        
        lf = ttk.Frame(nb, padding=5)
        nb.add(lf, text="📝 JSON")
        self.log_area = scrolledtext.ScrolledText(lf, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED)
        self.log_area.pack(fill=tk.BOTH, expand=True)
        
        rf = ttk.Frame(nb, padding=5)
        nb.add(rf, text="📊 Отчёт")
        self.report_area = scrolledtext.ScrolledText(rf, wrap=tk.WORD, font=("Arial", 10), state=tk.DISABLED)
        self.report_area.pack(fill=tk.BOTH, expand=True)
    
    def _new_interview(self):
        dlg = SetupDialog(self.root)
        self.root.wait_window(dlg.top)
        if dlg.result:
            self.orchestrator = InterviewOrchestrator(smart_mode=self.smart_var.get())
            self.orchestrator.set_model(AVAILABLE_MODELS.get(self.model_var.get(), "gemini-3-flash-preview"))
            self.orchestrator.start_session(Candidate(**dlg.result))
            self._clear_all()
            self._enable_chat()
            self.status_var.set("Начинаем...")
            self._run_async(self._greet())
    
    async def _greet(self):
        r = await self.orchestrator.generate_greeting()
        if "error" not in r:
            self._chat("agent", r["message"])
            self._thoughts(r.get("thoughts", []), r.get("turn_id", 0))
            self._diff(r.get("difficulty", 2))
            self._log()
            self.status_var.set("Ваш ход!")
    
    def _send(self):
        msg = self.input_entry.get().strip()
        if not msg:
            return
        self.input_entry.delete(0, tk.END)
        self._chat("user", msg)
        self.status_var.set("Обработка...")
        self._disable_input()
        self._run_async(self._process(msg))
    
    async def _process(self, msg: str):
        r = await self.orchestrator.process_message(msg)
        if r.get("finished"):
            self._finish(r)
        elif "error" not in r:
            self._chat("agent", r["message"])
            self._thoughts(r.get("thoughts", []), r.get("turn_id", 0))
            self._diff(r.get("difficulty", 2))
            self._log()
            self.status_var.set(f"Флаги: {', '.join(r.get('flags', []))}" if r.get("flags") else "Ваш ход!")
        self._enable_input()
    
    def _stop_interview(self):
        if self.orchestrator and self.orchestrator.session:
            self.status_var.set("Генерация...")
            self._disable_input()
            self._run_async(self._do_finish())
    
    async def _do_finish(self):
        self._finish(await self.orchestrator.finish_interview())
    
    def _finish(self, r: Dict):
        self._disable_chat()
        self._chat("system", "\n" + "="*50 + "\nИНТЕРВЬЮ ЗАВЕРШЕНО\n" + "="*50)
        if "feedback" in r:
            self._report(r["feedback"])
        self._log()
        self.status_var.set("Завершено")
    
    def _save_log(self):
        if not self.orchestrator or not self.orchestrator.session:
            messagebox.showwarning("Внимание", "Нет сессии")
            return
        fn = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")],
                                          initialfile=f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        if fn:
            base = fn[:-5] if fn.endswith(".json") else fn
            full = self.orchestrator.session.to_dict()
            with open(f"{base}_my_log.json", 'w', encoding='utf-8') as f:
                json.dump(full, f, ensure_ascii=False, indent=2)
            with open(f"{base}_log_formatted.json", 'w', encoding='utf-8') as f:
                json.dump(adapt_log_to_tz_format(full), f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Готово", f"Сохранено:\n• {base}_my_log.json\n• {base}_log_formatted.json")
    
    def _chat(self, role: str, text: str):
        self.chat_area.configure(state=tk.NORMAL)
        prefix = {"user": "👤 Вы: ", "agent": "🤖 Интервьюер: ", "system": "ℹ️ "}.get(role, "")
        tag = {"user": "user", "agent": "agent"}.get(role, "system")
        self.chat_area.insert(tk.END, f"\n{prefix}", tag)
        self.chat_area.insert(tk.END, f"{text}\n")
        self.chat_area.see(tk.END)
        self.chat_area.configure(state=tk.DISABLED)
    
    def _thoughts(self, thoughts: List[Dict], turn_id: int):
        self.thoughts_area.configure(state=tk.NORMAL)
        self.thoughts_area.insert(tk.END, f"\n{'─'*40}\nХод #{turn_id}\n")
        for t in thoughts:
            self.thoughts_area.insert(tk.END, f"[{t.get('agent','')}]: {t.get('thought','')}\n")
        self.thoughts_area.see(tk.END)
        self.thoughts_area.configure(state=tk.DISABLED)
    
    def _log(self):
        self.log_area.configure(state=tk.NORMAL)
        self.log_area.delete(1.0, tk.END)
        self.log_area.insert(tk.END, self.orchestrator.get_log_json())
        self.log_area.configure(state=tk.DISABLED)
    
    def _diff(self, lvl: int):
        self.diff_var.set(f"Сложность: {'⭐'*lvl}{'☆'*(5-lvl)}")
    
    def _report(self, fb: Dict):
        self.report_area.configure(state=tk.NORMAL)
        self.report_area.delete(1.0, tk.END)
        dec, tech, soft, rm = fb.get("decision",{}), fb.get("technical_review",{}), fb.get("soft_skills_review",{}), fb.get("roadmap",{})
        rec = dec.get("hiring_recommendation","")
        emoji = {"Strong Hire":"🌟🌟🌟","Hire":"✅","Maybe":"🤔","No Hire":"❌","Strong No Hire":"🚫"}.get(rec,"")
        
        txt = f'''{'='*60}\n                    ФИНАЛЬНЫЙ ОТЧЁТ\n{'='*60}\n
ВЕРДИКТ\n{'─'*35}\nГрейд: {dec.get('evaluated_grade','N/A')}\nРекомендация: {emoji} {rec}\nУверенность: {dec.get('confidence_score',0)}%\n{dec.get('explanation','')}\n
ТЕХНИЧЕСКИЕ НАВЫКИ ({tech.get('overall_score','N/A')}/10)\n{'─'*35}\n'''
        for s in tech.get("confirmed_skills", []):
            txt += f"✅ {s.get('topic','')} ({s.get('score','')}/10)\n"
        txt += "\nПробелы:\n"
        for g in tech.get("knowledge_gaps", []):
            txt += f"❌ {g.get('topic','')} [{g.get('severity','')}]\n"
        
        txt += f'''\nSOFT SKILLS\n{'─'*35}\nЯсность: {soft.get('clarity',{}).get('score','N/A')}/10\nЧестность: {soft.get('honesty',{}).get('score','N/A')}/10\nВовлечённость: {soft.get('engagement',{}).get('score','N/A')}/10\n
ПЛАН РАЗВИТИЯ\n{'─'*35}\n'''
        for item in rm.get("priority_topics", []):
            txt += f"📚 {item.get('topic','')} [{item.get('priority','')}]\n"
            for r in item.get("resources", []):
                txt += f"   🔗 {r}\n"
        
        txt += f'''\nФЛАГИ\n{'─'*35}\n🔴 {', '.join(fb.get('red_flags',[])) or 'нет'}\n🟢 {', '.join(fb.get('green_flags',[])) or 'нет'}\n
РЕЗЮМЕ: {fb.get('summary','')}\n{'='*60}'''
        self.report_area.insert(tk.END, txt)
        self.report_area.configure(state=tk.DISABLED)
    
    def _clear_all(self):
        for a in [self.chat_area, self.thoughts_area, self.log_area, self.report_area]:
            a.configure(state=tk.NORMAL)
            a.delete(1.0, tk.END)
            a.configure(state=tk.DISABLED)
    
    def _enable_chat(self):
        self.input_entry.configure(state=tk.NORMAL)
        self.send_btn.configure(state=tk.NORMAL)
    
    def _disable_chat(self):
        self.input_entry.configure(state=tk.DISABLED)
        self.send_btn.configure(state=tk.DISABLED)
    
    def _enable_input(self):
        self.input_entry.configure(state=tk.NORMAL)
        self.send_btn.configure(state=tk.NORMAL)
        self.input_entry.focus()
    
    def _disable_input(self):
        self.input_entry.configure(state=tk.DISABLED)
        self.send_btn.configure(state=tk.DISABLED)
    
    def _run_async(self, coro):
        # tkinter не дружит с async, поэтому через thread
        def run():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(coro)
        t = threading.Thread(target=run)
        t.start()
        def check():
            if t.is_alive():
                self.root.after(100, check)
        check()
    
    def run(self):
        self.root.mainloop()
        try:
            self.loop.run_until_complete(self.orchestrator.close())
        except:
            pass

class SetupDialog:
    def __init__(self, parent):
        self.result = None
        self.top = tk.Toplevel(parent)
        self.top.title("Новое интервью")
        self.top.geometry("400x320")
        self.top.resizable(False, False)
        self.top.transient(parent)
        self.top.grab_set()
        
        f = ttk.Frame(self.top, padding=20)
        f.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(f, text="Имя:").pack(anchor=tk.W)
        self.name = tk.StringVar(value="Алекс")
        ttk.Entry(f, textvariable=self.name, width=40).pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(f, text="Позиция:").pack(anchor=tk.W)
        self.pos = tk.StringVar(value="Backend Developer")
        ttk.Combobox(f, textvariable=self.pos, values=["Backend Developer","Frontend Developer","Fullstack Developer","QA Engineer","DevOps Engineer"], width=38).pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(f, text="Уровень:").pack(anchor=tk.W)
        self.grade = tk.StringVar(value="Junior")
        ttk.Combobox(f, textvariable=self.grade, values=["Junior","Middle","Senior","Lead"], width=38).pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(f, text="Опыт:").pack(anchor=tk.W)
        self.exp = tk.StringVar(value="Пет-проекты на Django")
        ttk.Entry(f, textvariable=self.exp, width=40).pack(fill=tk.X, pady=(0,20))
        
        bf = ttk.Frame(f)
        bf.pack(fill=tk.X)
        ttk.Button(bf, text="Начать", command=self._ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="Отмена", command=self.top.destroy).pack(side=tk.LEFT)
    
    def _ok(self):
        self.result = {"name": self.name.get() or "Кандидат", "position": self.pos.get() or "Backend Developer",
                       "grade": self.grade.get() or "Junior", "experience": self.exp.get() or "не указано"}
        self.top.destroy()

async def run_cli():
    print("="*55 + "\n   ТРЕНАЖЁР СОБЕСЕДОВАНИЙ\n" + "="*55)
    name = input("Имя: ").strip() or "Кандидат"
    position = input("Позиция: ").strip() or "Backend Developer"
    grade = input("Уровень: ").strip() or "Junior"
    exp = input("Опыт: ").strip() or "пет-проекты"
    smart = input("Smart Mode? (y/n): ").strip().lower() == 'y'
    
    orch = InterviewOrchestrator(smart_mode=smart)
    orch.start_session(Candidate(name, position, grade, exp))
    
    print("\n" + "-"*55 + "\nИнтервью началось! Команды: стоп, фидбэк\n" + "-"*55)
    r = await orch.generate_greeting()
    print(f"🤖 {r['message']}\n")
    
    while True:
        try:
            inp = input("👤 Вы: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not inp:
            continue
        
        r = await orch.process_message(inp)
        if r.get("finished"):
            print("\n" + "="*55 + "\n   ЗАВЕРШЕНО\n" + "="*55)
            fb = r["feedback"]
            dec = fb.get("decision", {})
            print(f"Грейд: {dec.get('evaluated_grade')}\nРекомендация: {dec.get('hiring_recommendation')}\n{fb.get('summary','')}")
            break
        print(f"\n🤖 {r['message']}\n")
    
    base = f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    full = orch.session.to_dict()
    with open(f"{base}_my_log.json", 'w', encoding='utf-8') as f:
        json.dump(full, f, ensure_ascii=False, indent=2)
    with open(f"{base}_log_formatted.json", 'w', encoding='utf-8') as f:
        json.dump(adapt_log_to_tz_format(full), f, ensure_ascii=False, indent=2)
    print(f"\n💾 Логи: {base}_my_log.json, {base}_log_formatted.json")
    await orch.close()
