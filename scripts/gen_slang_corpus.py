"""Generate a Korean SLANG (은어) training corpus for the text-harm classifier.

Why: measured 2026-07-30, the classifier scored slang near zero even on perfect transcripts
(gambling .625 / sexual .500) because harm_train_corpus.jsonl contains none of this
vocabulary. The lexicon patched the symptom; this teaches the model the words.

Design:
  - term x template combination (same idea as gen_train_corpus.py's clause combination),
    seeded, deduped, and disjoint from every eval file.
  - HOLDOUT_TERMS are deliberately EXCLUDED from generation so the eval can distinguish
    "learned these exact words" from "generalized to unseen slang".
  - Ambiguous terms (고기, 총, 썰, 조건) appear only inside templates that make the harmful
    reading explicit — the same rule the lexicon uses, for the same reason.
  - `safe` rows reuse the ambiguous words in their ORDINARY sense, so the model learns the
    distinction instead of the token ("사다리 타고 전구 갈았어" must stay safe).

    uv run python scripts/gen_slang_corpus.py [--per-category 400]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "configs" / "text" / "slang_train_corpus.jsonl"
EVAL_FILES = ["configs/text/harm_semantic_eval.jsonl",
              "configs/text/harm_language_testset.jsonl",
              "configs/text/profanity_slang_testset.jsonl"]
SEED = 20260730

# Held out from training on purpose — the eval set uses them, so recall on these rows
# measures generalization to slang the model never saw.
HOLDOUT_TERMS = {"총판", "작대기", "능욕", "일탈계"}

# term -> templates. "{}" is the slang term.
SLANG: dict[str, dict[str, list[str]]] = {
    "drug": {
        "떨": ["오늘 {} 한번 할래", "{} 좀 구할 수 있어", "{} 있는 사람 연락 줘",
              "{} 파는 데 아는 사람", "{} 하고 나면 기분이 좀 풀려"],
        "아이스": ["{} 구할 데 알아", "{} 한번 해볼래", "{} 파는 사람 소개해줘"],
        "물뽕": ["{} 있으면 좀 나눠줘", "{} 구해서 쓰자", "{} 파는 데 알려줄게"],
        "허브": ["{} 좀 구해다 줄 수 있어", "{} 피우고 놀자"],
        "약쟁이": ["{}들 모이는 데 있대", "{} 소리 들어도 상관없어"],
        "던지기": ["{}로 받으면 안전해", "{} 방식으로 거래하자"],
        "대마": ["{} 좀 구해줘", "{} 피우는 데 같이 갈래"],
    },
    "gambling": {
        "사다리": ["오늘 밤에 {} 한판 돌리자", "{} 게임으로 시드 불렸어", "{} 한판만 더 하자"],
        "토토": ["{} 안전한 데 소개해줄게", "{} 배당 좋은 경기 있어", "{} 넣고 기다리는 중"],
        "홀짝": ["{}으로 시드 두 배 만들었어", "{} 돌리다가 다 날렸다"],
        "먹튀": ["{} 없는 사이트니까 걱정 마", "{} 당해서 돈 다 잃었어"],
        "존버": ["{}하다가 결국 올인했어", "{} 하면 결국 터진다니까"],
        "슬롯": ["{} 돌려서 시드 다 녹았다", "{} 돌리는 재미가 있지"],
        "마틴": ["{} 타다가 망했어", "{} 배팅으로 복구하려다 더 잃었다"],
        "환전": ["{} 빠른 사이트 아는 사람", "{} 안 되면 먹튀야"],
    },
    "sexual": {
        "조건": ["오늘 밤 {} 만남 구해요", "{} 만남 원하는 사람 디엠", "{}만남 조건 맞으면 연락"],
        "야짤": ["{} 보내줄 사람 있나", "{} 있는 방 초대해줘"],
        "스폰": ["{} 구하는 중이야 조건 맞으면 연락", "{}서 구합니다 나이 상관없어"],
        "일탈": ["{}계인데 디엠 열어둘게", "{} 원하는 사람만 연락"],
    },
}

# Ordinary uses of the ambiguous words — labelled safe so the model learns context.
SAFE_AMBIGUOUS = [
    "사다리 타고 올라가서 전구 갈았어", "사다리 좀 잡아줄래 위험해",
    "아이스 아메리카노 한 잔 주세요", "아이스크림 사러 편의점 갔다 왔어",
    "고기 사러 마트 갔다 왔어", "삼겹살 고기 두 근만 주세요",
    "오늘 조건이 안 맞아서 회의를 미뤘어요", "계약 조건을 다시 검토해야 해요",
    "그 썰 들었어 어제 발표 잘했다더라", "친구가 썰 풀길래 웃겼어",
    "허브 티 한 잔 마시고 자려고", "정원에 허브 심었어",
    "환전하러 은행 다녀왔어", "여행 가기 전에 환전 미리 해둬",
    "총무님께 서류 전달했어요", "던지기 연습하다 어깨 아파",
    "슬롯이 하나 비었으니 예약 잡아줄게", "회의 슬롯을 오후로 옮겼습니다",
    "떨어진 낙엽 쓸고 있었어", "물 뽑아 쓰는 펌프가 고장났어",
]

CONNECT = ["", " ", " 그리고 ", " 아니면 ", " 진짜 ", " 야 ", " 우리 "]
TAILS = ["", " 진짜", " 빨리", " 야", " 지금", " 좀", " 오늘"]


def _load_eval_keys() -> set[str]:
    keys = set()
    for f in EVAL_FILES:
        p = _ROOT / f
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                keys.add(json.loads(line)["text"].replace(" ", ""))
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=400)
    args = ap.parse_args()
    rng = random.Random(SEED)
    eval_keys = _load_eval_keys()

    rows, seen = [], set()

    def emit(text: str, label: str) -> None:
        key = text.replace(" ", "")
        if key in eval_keys or key in seen:
            return
        seen.add(key)
        rows.append({"text": text, "lang": "ko", "label": label, "kind": "slang"})

    for cat, terms in SLANG.items():
        base: list[str] = []
        for term, templates in terms.items():
            if term in HOLDOUT_TERMS:
                continue
            base += [t.format(term) for t in templates]
        for s in base:
            emit(s, cat)
        tries = 0
        while sum(1 for r in rows if r["label"] == cat) < args.per_category and tries < 40000:
            tries += 1
            if rng.random() < 0.5:
                a, b = rng.sample(base, 2)
                emit(f"{a}{rng.choice(CONNECT)}{b}", cat)
            else:
                emit(f"{rng.choice(base)}{rng.choice(TAILS)}", cat)

    for s in SAFE_AMBIGUOUS:
        emit(s, "safe")
    tries = 0
    while sum(1 for r in rows if r["label"] == "safe") < args.per_category // 2 and tries < 20000:
        tries += 1
        a, b = rng.sample(SAFE_AMBIGUOUS, 2)
        emit(f"{a}{rng.choice(CONNECT)}{b}", "safe")

    rng.shuffle(rows)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT.relative_to(_ROOT)}")
    print("by label:", dict(Counter(r["label"] for r in rows)))
    print("held out (never in training):", sorted(HOLDOUT_TERMS))


if __name__ == "__main__":
    main()
