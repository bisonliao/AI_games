from __future__ import annotations

import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOT = ROOT / "content"


BEGINNER_KEYS = [
    ["q", "a", "z"],
    ["w", "s", "x"],
    ["e", "d", "c"],
    ["r", "t", "f", "g", "v", "b"],
    ["f", "j", " "],
    ["y", "u", "h", "j", "n", "m"],
    ["i", "k", ","],
    ["o", "l", "."],
    ["p", ";", "/"],
]


ENGLISH_SENTENCES = [
    "A careful typist keeps both hands relaxed and watches the screen.",
    "The small desk lamp makes every key easy to find in the evening.",
    "Practice begins slowly, but steady fingers grow faster each week.",
    "When a mistake appears, keep moving and finish the whole sentence.",
    "The blue notebook, the sharp pencil, and the quiet room are ready.",
    "A bright kite crossed the park while the children counted clouds.",
    "Good habits are built from short lessons repeated with patience.",
    "The keyboard has many keys, but each finger has a simple home.",
    "After school, we packed the books and walked home before dinner.",
    "Clear typing needs calm eyes, gentle hands, and a little courage.",
    "The garden path was narrow, so everyone walked slowly and smiled.",
    "A friendly teacher wrote a short story on the board before lunch.",
    "Please read the line first, then type it without rushing ahead.",
    "The toy train rolled across the floor and stopped beside the chair.",
    "Rain tapped on the window while warm light filled the study room.",
    "Some words are tiny, some are long, and all of them deserve care.",
    "The clock ticked softly as the family prepared a simple breakfast.",
    "Typing is easier when the shoulders are loose and the wrists rest.",
    "Each new lesson adds a few more keys to the map inside your hands.",
    "A clean screen and a comfortable chair make practice feel friendly.",
    "The red ball bounced under the table and rolled near the bookshelf.",
    "Before starting again, take one deep breath and place your fingers.",
    "Speed can wait; accuracy is the strong root of confident typing.",
    "The little robot sorted numbers, labels, colors, and tiny boxes.",
    "Every finished line is a small step toward easier computer work.",
    "A quiet afternoon is a fine time to learn a useful skill.",
    "The paper boat moved along the stream and passed three smooth stones.",
    "Warm soup, fresh bread, and a clean spoon waited on the table.",
    "The class listened carefully while the music began to play.",
    "Practice turns strange keys into familiar places for your fingers.",
]


CHINESE_SENTENCES = [
    "清晨的阳光照在书桌上，键盘上的字母一排一排地亮起来。",
    "孩子把双手轻轻放好，先看清目标文字，再慢慢敲下每一个键。",
    "练习打字不需要着急，重要的是保持坐姿端正，眼睛看屏幕。",
    "如果敲错了也不用慌，因为这次练习会继续向前，下一次就会更熟悉。",
    "窗外有风吹过树梢，屋里很安静，只有键盘发出轻轻的声音。",
    "一段文字打完以后，可以看看正确率，也可以休息一下再开始新的练习。",
    "中文输入需要先想好词语，再通过输入法选择合适的字。",
    "每天练一点点，手指会越来越灵活，看到文字时也会更快找到对应的按键。",
    "爸爸妈妈可以在旁边陪着孩子，把练习变成轻松的小任务。",
    "当速度变快以后，仍然要记得准确最重要，稳稳地输入比匆忙地出错更有价值。",
    "春天的校园里有很多新的叶子，走廊尽头传来同学们的笑声。",
    "小朋友把今天的计划写在纸上：读书、练字、打字，然后出去运动。",
    "电脑是一种有用的工具，学会输入文字以后，就能记录想法和故事。",
    "练习时可以把目标分成一小段一小段，完成以后再给自己一个鼓励。",
    "输入中文时，眼睛要看清候选词，手指也要记得回到熟悉的位置。",
    "周末的下午，家里飘着饭菜的香味，窗台上的植物安静地长高。",
    "故事里的小船穿过桥洞，河水映着天空，也映着岸边的灯。",
    "每一次练习都不是考试，而是在帮助大脑和手指慢慢建立联系。",
    "如果今天只进步了一点点，也值得开心，因为好习惯就是这样长出来的。",
    "打字练熟以后，写作业、查资料、和家人分享想法都会方便许多。",
    "夜晚来临时，书桌被整理得干干净净，第二天的练习也准备好了。",
    "孩子认真看着屏幕，发现自己已经能连续输入很多正确的字。",
    "学习新技能像走一段小路，开始有些陌生，走多了就会越来越顺。",
    "键盘上的每个按键都有自己的位置，熟悉以后就像认识了新朋友。",
    "慢慢来，稳稳打，先把准确率练好，速度自然会在以后跟上来。",
]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def beginner_text(keys: list[str], seed: int) -> str:
    random.seed(seed)
    chunks: list[str] = []
    for _ in range(6):
        chars: list[str] = []
        for i in range(26):
            if keys == ["f", "j", " "]:
                chars.append(" " if i % 2 else random.choice(["f", "j"]))
            else:
                chars.append(random.choice(keys))
            if i in {5, 12, 19}:
                chars.append(" ")
        chunks.append("".join(chars).strip())
    return " ".join(chunks)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


def english_text(seed: int) -> str:
    random.seed(seed)
    pool = ENGLISH_SENTENCES[:]
    random.shuffle(pool)
    picked: list[str] = []
    count = 0
    while count < 95:
        if not pool:
            pool = ENGLISH_SENTENCES[:]
            random.shuffle(pool)
        sentence = pool.pop()
        picked.append(sentence)
        count += word_count(sentence)
    return " ".join(picked)


def chinese_char_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def chinese_text(seed: int) -> str:
    random.seed(seed)
    pool = CHINESE_SENTENCES[:]
    random.shuffle(pool)
    picked: list[str] = []
    count = 0
    while count < 290:
        if not pool:
            pool = CHINESE_SENTENCES[:]
            random.shuffle(pool)
        sentence = pool.pop()
        picked.append(sentence)
        count += chinese_char_count(sentence)
    return "".join(picked)


def main() -> None:
    for lesson_index, keys in enumerate(BEGINNER_KEYS, start=1):
        for item_index in range(1, 6):
            path = CONTENT_ROOT / "beginner" / f"lesson_{lesson_index:02d}" / f"{item_index:02d}.txt"
            write_text(path, beginner_text(keys, lesson_index * 100 + item_index))

    for item_index in range(1, 21):
        write_text(CONTENT_ROOT / "intermediate" / f"{item_index:02d}.txt", english_text(1000 + item_index))
        write_text(CONTENT_ROOT / "advanced" / f"{item_index:02d}.txt", chinese_text(2000 + item_index))


if __name__ == "__main__":
    main()
