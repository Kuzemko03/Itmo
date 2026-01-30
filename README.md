## README.md

# AI Interview Trainer

Мультиагентная система для тренировки технических собеседований.

## Установка

1. Клонировать репозиторий:
bash
git clone https://github.com/Kuzemko03/Itmo.git
cd ai-interview-trainer

2. Установить зависимости:
bash
pip install -r requirements.txt


3. Заполнить и настроить файл secrets.json 
Если есть прокси:

{
    "GEMINI_API_KEY": "ваш_ключ",
    "PROXY": "http://логин:пароль@хост:порт"
}

Если прокси нет — используйте VPN:
Включите VPN (на весь трафик системы, не только браузер)
В secrets.json укажите:
{
    "GEMINI_API_KEY": "ваш_ключ",
    "PROXY": null
}


## Запуск

bash
python main.py


## Использование

0. Включить кнопку "Smart", с ней размышления лучше, так как включается еще один агент
1. Ввести имя кандидата
2. Выбрать позицию (Backend Developer, Frontend Developer и др.)
3. Выбрать грейд (Junior, Middle, Senior)
4. Отвечать на вопросы интервьюера
5. Написать "стоп" для завершения
6. Получить фидбек и лог в `interview_log.json`

## Агенты

| Агент | Функция |
|-------|---------|
| Observer | Анализ ответов кандидата |
| Interviewer | Генерация вопросов |
| DifficultyController | Адаптивная сложность |
| FactChecker | Детекция галлюцинаций |
| ContradictionDetector | Поиск противоречий |
| DepthProber | Оценка глубины знаний |
| MetaReviewer | Контроль качества диалога |
| Evaluator | Финальный отчёт |

## Модель
Самая быстрая и нормально работающая
`google/gemini-2.0-flash`

Для лучших размышлений выбирать gemini 3 flash/pro в интерфейсе

## Дополнительные файлы

test_runner.py писал для себя для тестов разных сценариев в автоматическом режиме
