"""Generate a larger, diverse KO/EN labeled test set for the language branch.

Hand-authored *clause banks* per (category, kind, language) — each clause is a complete,
grammatical sentence — are sampled and lightly joined with connectors to multiply
diversity WITHOUT the grammatical breakage that word-slot templating causes in Korean.
Deterministic (seeded). Output: configs/text/harm_language_testset.jsonl.

Purpose: the 89-row set is too small for stable per-cell metrics; this scales to a few
hundred so the end-to-end (audio-first) numbers — especially Korean — have tighter CIs.
Honest limit: still synthetic (my authoring), so it probes generalization within my
distribution; real transcripts remain the final validation.

    uv run python scripts/gen_language_testset.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "configs" / "text" / "harm_language_testset.jsonl"
SEED = 20260712
PER_CELL = 22   # target rows per (category, kind, lang) cell

# Each entry: complete grammatical clauses.
# Kind: explicit/implicit (harm), plain/tricky/stance (safe).
BANK: dict = {
    ("threat", "explicit", "ko"): [
        "너 오늘 내 손에 죽는다", "가만 안 둘 테니 각오해", "칼 들고 찾아갈 거야",
        "패 죽여버리기 전에 사라져", "네 다리를 분질러 놓겠어", "묻어버릴 줄 알아",
        "다시는 못 일어나게 밟아줄게", "너희 집 어딘지 다 알아 곧 간다",
        "손봐줄 테니 기다려", "죽고 싶지 않으면 꺼져", "반쯤 죽여놓을 거야",
        "머리통을 부숴버리겠어", "오늘이 네 마지막 날이야",
    ],
    ("threat", "implicit", "ko"): [
        "밤길 조심하는 게 좋을 거야", "네가 무슨 짓 했는지 곧 알게 될걸",
        "이대로 끝날 거라 생각하지 마", "네 뒤를 계속 보고 있을 거야",
        "가족들 무사하길 바라면 잘 생각해", "곧 크게 후회할 일이 생길 거야",
        "내가 어떤 사람인지 아직 모르나 보네", "그 입 함부로 놀리면 어떻게 되는지 알지",
        "조만간 아주 안 좋은 일이 생길 수도 있어", "네가 한 짓 그대로 돌려받게 될 거야",
        "세상 참 무서운 곳이야 조심해", "혼자 다니지 않는 게 신상에 좋아",
    ],
    ("threat", "explicit", "en"): [
        "I will kill you tonight", "I'm going to break your legs",
        "you're a dead man if you show up", "I'll hunt you down and end you",
        "I swear I'll make you bleed", "stay quiet or I'll bury you",
        "I'll smash your face in", "keep talking and you won't walk again",
        "I know where you live and I'm coming", "you won't see tomorrow",
    ],
    ("threat", "implicit", "en"): [
        "you'd better watch your back from now on", "accidents happen to people who talk",
        "it'd be a shame if something happened to your shop",
        "I wouldn't go home alone if I were you", "you have no idea what's coming",
        "you'll regret ever crossing me", "be careful out there, things happen",
        "I hope your family stays safe, think about it", "you brought this on yourself",
        "some people just disappear, you know",
    ],
    ("sexual", "explicit", "ko"): [
        "나랑 지금 자자", "옷 다 벗고 기다려", "야한 사진 보내줘",
        "가슴 만지게 해줘", "성관계 하고 싶어 지금", "네 몸 구석구석 보고 싶어",
        "우리 은밀하게 즐기자", "속옷만 입고 찍어서 보내",
    ],
    ("sexual", "implicit", "ko"): [
        "오늘 밤 둘만 특별한 시간 보내자", "몸매가 예술이라 참기 힘드네",
        "이리 와서 좀 더 가까이 안겨봐", "우리 집에 아무도 없는데 올래",
        "네 입술에서 눈을 못 떼겠어", "불 끄고 조용히 둘이서만",
    ],
    ("sexual", "explicit", "en"): [
        "let's have sex tonight", "send me a nude photo now", "take off your clothes for me",
        "I want to touch your body", "come to my place and strip", "I want you naked right now",
    ],
    ("sexual", "implicit", "en"): [
        "why don't you slip into something less comfortable", "nobody's home, come over",
        "I can't stop staring at your body", "let's turn off the lights and get close",
        "spend the night and let's not sleep", "you and me, somewhere private, tonight",
    ],
    ("gambling", "explicit", "ko"): [
        "카지노 가서 크게 걸자", "이번 판 올인이다", "슬롯에 월급 다 넣었어",
        "포커 판돈 더 올려", "잭팟 노리고 다 걸었어", "룰렛에 전부 걸어",
        "판돈 두 배로 올려 콜", "오늘 밤 도박장 가자",
    ],
    ("gambling", "implicit", "ko"): [
        "이 한 방이면 인생 역전이야 다 걸어", "오늘 느낌 좋다 크게 지른다",
        "본전 생각 말고 끝까지 가보자", "떨어진 만큼 한 번에 복구하면 돼",
        "질 것 같으면 두 배로 걸어서 만회해", "마지막으로 한 번만 더 걸어보자",
    ],
    ("gambling", "explicit", "en"): [
        "let's hit the casino and bet big", "I'm going all in this hand",
        "put it all on red", "double down on blackjack", "I bet my whole paycheck on it",
        "spin the roulette, max bet",
    ],
    ("gambling", "implicit", "en"): [
        "one big bet and we're set for life", "I'm feeling lucky, let's risk it all",
        "double or nothing, come on", "we can win it all back in one hand",
        "just one more round to break even", "go big or go home, bet everything",
    ],
    ("drug", "explicit", "ko"): [
        "필로폰 좀 구해줘", "대마초 같이 피우자", "코카인 어디서 사",
        "약 한 판 하자", "히로뽕 좀 구할 수 있어", "엑스터시 오늘 밤에 하자",
        "마약 팔 사람 아는데 연결해줄까", "주사기 챙겨서 한 대 맞자",
    ],
    ("drug", "implicit", "ko"): [
        "좋은 물건 새로 들어왔는데 해볼래", "그거 한 대 빨면 세상이 달라 보여",
        "기분 뜨는 거 하나 있는데 같이 할래", "이거 하면 밤새 안 자도 돼",
        "특별한 걸로 준비했으니까 와", "한 번만 맛보면 계속 찾게 돼",
    ],
    ("drug", "explicit", "en"): [
        "where can I buy some meth", "let's smoke weed tonight", "score me some cocaine",
        "let's do a line right now", "I need to shoot up, got a needle",
        "he's dealing pills on the corner",
    ],
    ("drug", "implicit", "en"): [
        "I got the good stuff if you want a hit", "one taste and you'll be hooked",
        "this'll keep you up all night, try it", "got something that'll make you fly",
        "fresh batch just came in, wanna try", "it'll take the edge off, just one hit",
    ],
    ("safe", "plain", "ko"): [
        "오늘 점심 뭐 먹을까", "주말에 등산 갈래", "회의를 3시로 미뤄도 될까요",
        "이 코드 리뷰 좀 부탁해", "엄마 생신 선물 골라야 하는데", "커피 한잔 하러 가자",
        "내일 비 온다니까 우산 챙겨", "발표 자료 거의 다 됐어", "새 카페 커피 맛있더라",
        "주차는 지하 2층에 하면 돼", "이 책 다 읽으면 빌려줄게", "저녁에 같이 산책할래",
        "시험 잘 보고 와", "고양이 밥 좀 챙겨줘", "택배 오면 문 앞에 놔달라고 해",
    ],
    ("safe", "tricky", "ko"): [
        "이 노래 진짜 죽인다 완전 최고", "그 배우 연기 죽여주더라",
        "인생 올인해서 공부만 했어", "시험 붙어서 잭팟 터진 기분이야",
        "카지노 로얄 그 영화 명작이지", "매운 라면 먹고 죽는 줄 알았네",
        "약속 시간 늦지 않게 서둘러", "너 죽도록 보고 싶었어",
        "숙제 끝내서 이제 좀 살 것 같다", "그 경기 손에 땀을 쥐게 하더라",
    ],
    ("safe", "stance", "ko"): [
        "마약은 절대 하면 안 되는 거야", "필로폰은 몸 망치는 나쁜 마약이야",
        "도박은 인생을 망치니까 하지 마", "폭력으로는 아무것도 해결 안 돼",
        "친구가 마약 끊고 재활에 성공했대", "학교에서 약물 예방 교육 받았어",
        "뽕 같은 거 할 생각조차 하면 안 돼", "도박 중독은 치료가 필요한 병이야",
    ],
    ("safe", "plain", "en"): [
        "what should we have for lunch", "let's go hiking this weekend",
        "can we move the meeting to 3", "could you review my pull request",
        "I need a gift for my mom's birthday", "let's grab a coffee",
        "bring an umbrella, it'll rain", "the slides are almost done",
        "that new cafe has great coffee", "park on level two",
        "I'll lend you the book when I'm done", "want to take a walk tonight",
        "good luck on your exam", "please feed the cat", "leave the package by the door",
    ],
    ("safe", "tricky", "en"): [
        "she absolutely killed that presentation", "I'd kill for a vacation right now",
        "that guitar solo was sick", "we hit the jackpot with this new hire",
        "Casino Royale is my favorite film", "I'm dead tired after that shift",
        "this workout is gonna be the death of me", "he's all in on the startup",
        "that joke slayed, so funny", "the deadline is killing me",
    ],
    ("safe", "stance", "en"): [
        "drugs ruin lives, never try them", "please don't gamble, it destroyed my uncle",
        "violence is never the answer", "he finally got clean after years",
        "our campaign teaches kids to say no to drugs", "cocaine is dangerous, stay away",
        "gambling addiction is a treatable illness", "meth will destroy your body, avoid it",
    ],
}

CONNECT = {"ko": ["", " 그리고 ", " 진짜 ", " 야 "], "en": ["", " and ", " seriously, ", " hey, "]}


def main() -> None:
    rng = random.Random(SEED)
    rows, seen = [], set()
    for (cat, kind, lang), clauses in BANK.items():
        cell = []
        base = list(clauses)
        rng.shuffle(base)
        for c in base:                       # every base clause once
            cell.append(c)
        while len(cell) < PER_CELL:          # multiply by joining two distinct clauses
            a, b = rng.sample(clauses, 2)
            conn = rng.choice(CONNECT[lang][1:])
            cell.append(f"{a}{conn}{b}")
        for text in cell[:PER_CELL]:
            key = text.strip()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "lang": lang, "label": cat, "kind": kind})
    rng.shuffle(rows)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT}")
    print("by label:", dict(Counter(r["label"] for r in rows)))
    print("by lang :", dict(Counter(r["lang"] for r in rows)))
    print("by kind :", dict(Counter(r["kind"] for r in rows)))


if __name__ == "__main__":
    main()
