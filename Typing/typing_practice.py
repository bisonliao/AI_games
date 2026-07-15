from __future__ import annotations

import json
import os
import random
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
CONTENT_ROOT = ROOT / "content"


BEGINNER_LESSONS = [
    {
        "id": 1,
        "title": "左手小拇指",
        "keys": ["q", "a", "z"],
        "hint": "重点练 q / a / z",
    },
    {
        "id": 2,
        "title": "左手无名指",
        "keys": ["w", "s", "x"],
        "hint": "重点练 w / s / x",
    },
    {
        "id": 3,
        "title": "左手中指",
        "keys": ["e", "d", "c"],
        "hint": "重点练 e / d / c",
    },
    {
        "id": 4,
        "title": "左手食指",
        "keys": ["r", "t", "f", "g", "v", "b"],
        "hint": "重点练 r / t / f / g / v / b",
    },
    {
        "id": 5,
        "title": "双手大拇指",
        "keys": [" "],
        "hint": "重点练空格键",
    },
    {
        "id": 6,
        "title": "右手食指",
        "keys": ["y", "u", "h", "j", "n", "m"],
        "hint": "重点练 y / u / h / j / n / m",
    },
    {
        "id": 7,
        "title": "右手中指",
        "keys": ["i", "k", ","],
        "hint": "重点练 i / k / ,",
    },
    {
        "id": 8,
        "title": "右手无名指",
        "keys": ["o", "l", "."],
        "hint": "重点练 o / l / .",
    },
    {
        "id": 9,
        "title": "右手小拇指",
        "keys": ["p", ";", "/"],
        "hint": "重点练 p / ; / /",
    },
]


ENGLISH_SENTENCES = [
    "A calm morning is a good time to learn a careful skill.",
    "The small lamp on the desk makes the keyboard easy to see.",
    "Every letter has a place, and every finger has a job.",
    "When a mistake happens, keep going and finish the line.",
    "Good typing is not magic; it grows from steady practice.",
    "The red kite moved above the trees while the children watched.",
    "I packed a blue pencil, a clean notebook, and a snack for school.",
    "Rain tapped on the window, but the room stayed warm and bright.",
    "A friendly robot counted the stars and wrote each number down.",
    "The garden path was narrow, so we walked slowly and carefully.",
    "Please read the sentence first, then type it with patient hands.",
    "Some words are short, some words are long, and all of them matter.",
    "The clock ticked softly as the cat slept beside the chair.",
    "After lunch, we fixed the toy train and tested it on the floor.",
    "Clear practice builds speed, accuracy, and quiet confidence.",
]


CHINESE_SENTENCES = [
    "清晨的阳光照在书桌上，键盘上的字母一排一排地亮起来。",
    "孩子把双手轻轻放好，先看清目标文字，再慢慢敲下每一个键。",
    "练习打字不需要着急，重要的是保持坐姿端正，眼睛看屏幕，手指回到正确的位置。",
    "如果敲错了也不用慌，因为这次练习会继续向前，下一次就会更熟悉。",
    "窗外有风吹过树梢，屋里很安静，只有键盘发出轻轻的声音。",
    "一段文字打完以后，可以看看正确率，也可以休息一下再开始新的练习。",
    "中文输入需要先想好词语，再通过输入法选择合适的字，这也是一种很好的专注训练。",
    "每天练一点点，手指会越来越灵活，看到文字时也会更快找到对应的按键。",
    "爸爸妈妈可以在旁边陪着孩子，把练习变成轻松的小任务，而不是紧张的考试。",
    "当速度变快以后，仍然要记得准确最重要，稳稳地输入比匆忙地出错更有价值。",
]


def json_response(handler: SimpleHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def beginner_lesson(lesson_id: int) -> dict | None:
    for lesson in BEGINNER_LESSONS:
        if lesson["id"] == lesson_id:
            return lesson
    return None


def previous_beginner_keys(lesson_id: int) -> list[str]:
    keys: list[str] = []
    for lesson in BEGINNER_LESSONS:
        if lesson["id"] <= lesson_id:
            keys.extend(lesson["keys"])
    return keys


def read_random_content(files: list[Path]) -> str | None:
    available = [path for path in files if path.is_file()]
    if not available:
        return None
    return random.choice(available).read_text(encoding="utf-8").strip()


def beginner_content_files(lesson_id: int) -> list[Path]:
    lesson_dir = CONTENT_ROOT / "beginner" / f"lesson_{lesson_id:02d}"
    return sorted(lesson_dir.glob("*.txt"))


def level_content_files(level: str) -> list[Path]:
    return sorted((CONTENT_ROOT / level).glob("*.txt"))


def generate_beginner_text(lesson_id: int, mode: str) -> str:
    lesson = beginner_lesson(lesson_id) or BEGINNER_LESSONS[0]
    focus_keys = lesson["keys"]
    review_keys = previous_beginner_keys(lesson["id"])
    use_review = mode == "review" and len(review_keys) > len(focus_keys)

    chunks: list[str] = []
    for _ in range(6):
        chars: list[str] = []
        for i in range(26):
            if lesson["id"] == 5:
                chars.append(" " if i % 2 else random.choice(["f", "j"]))
            elif use_review and random.random() < 0.3:
                chars.append(random.choice(review_keys))
            else:
                chars.append(random.choice(focus_keys))

            if i in {5, 12, 19}:
                chars.append(" ")
        chunks.append("".join(chars).strip())
    return " ".join(chunks)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


def generate_english_text() -> str:
    sentences = ENGLISH_SENTENCES[:]
    random.shuffle(sentences)
    picked: list[str] = []
    count = 0
    while count < 95:
        if not sentences:
            sentences = ENGLISH_SENTENCES[:]
            random.shuffle(sentences)
        sentence = sentences.pop()
        picked.append(sentence)
        count += word_count(sentence)
    return " ".join(picked)


def chinese_char_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def generate_chinese_text() -> str:
    sentences = CHINESE_SENTENCES[:]
    random.shuffle(sentences)
    picked: list[str] = []
    count = 0
    while count < 290:
        if not sentences:
            sentences = CHINESE_SENTENCES[:]
            random.shuffle(sentences)
        sentence = sentences.pop()
        picked.append(sentence)
        count += chinese_char_count(sentence)
    return "".join(picked)


def exercise_payload(query: dict[str, list[str]]) -> dict:
    level = query.get("level", ["beginner"])[0]

    if level == "beginner":
        lesson_id = int(query.get("lesson", ["1"])[0])
        lesson = beginner_lesson(lesson_id) or BEGINNER_LESSONS[0]
        text = read_random_content(beginner_content_files(lesson["id"]))
        return {
            "level": level,
            "title": f"初阶第 {lesson['id']} 课：{lesson['title']}",
            "subtitle": lesson["hint"],
            "text": text or generate_beginner_text(lesson["id"], "focus"),
            "lineLength": 34,
        }

    if level == "intermediate":
        text = read_random_content(level_content_files("intermediate"))
        return {
            "level": level,
            "title": "中阶：英文文章练习",
            "subtitle": "约 100 个英文单词，包含常见标点。",
            "text": text or generate_english_text(),
            "lineLength": 68,
        }

    if level == "advanced":
        text = read_random_content(level_content_files("advanced"))
        return {
            "level": level,
            "title": "高阶：中文文章练习",
            "subtitle": "约 300 个汉字，包含中文标点。",
            "text": text or generate_chinese_text(),
            "lineLength": 24,
        }

    return {"error": "unknown level"}


class TypingHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/lessons":
            json_response(self, {"lessons": BEGINNER_LESSONS})
            return
        if parsed.path == "/api/exercise":
            payload = exercise_payload(parse_qs(parsed.query))
            status = 400 if "error" in payload else 200
            json_response(self, payload, status)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def log_message(self, format: str, *args) -> None:
        print("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format % args))


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), TypingHandler)
    print(f"Typing practice server is running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
