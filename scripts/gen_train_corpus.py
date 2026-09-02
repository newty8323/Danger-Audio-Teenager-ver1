"""Generate a research-scale synthetic training corpus for the text-harm classifier.

Larger, diverse clause banks per (category, language, explicit/implicit) are combined
pairwise (with connectors) to multiply into thousands per category — the scale comparable
work uses (Jigsaw ~160k, K-MHaS ~109k, Unsmile ~18k). Deterministic (seeded). Output:
configs/text/harm_train_corpus.jsonl. Kept DISJOINT from the eval sets (dedup).

Honest limit: still synthetic (author distribution), so diversity is bounded; real benign
negatives (kor_unsmile clean) are added at train time, and eval stays on real held-out data.

    uv run python scripts/gen_train_corpus.py [--per-category 1800]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "configs" / "text" / "harm_train_corpus.jsonl"
SEED = 991

# ~40 diverse base clauses per harmful category (KO+EN, implicit+explicit), each grammatical.
BANK: dict[str, list[str]] = {
 "threat": [
  "네가 한 짓 절대 안 잊는다", "우리 다시 마주칠 날이 반드시 온다", "내가 누군지 곧 뼈저리게 알게 될 거다",
  "밤길 조심하는 게 좋을 거다", "그 대가는 반드시 치르게 만들어주지", "너희 식구들도 마음 편치 못할 거다",
  "지금 웃을 수 있는 것도 얼마 안 남았어", "어디 한번 끝까지 가보자", "네 인생 이제부터 꼬이기 시작할 거다",
  "오늘 일 두고두고 후회하게 해주마", "곱게 넘어갈 거라는 착각은 버려", "언젠가 골목에서 딱 마주치겠지",
  "너 오늘 내 손에 죽는다", "가만 안 둘 테니 각오해라", "칼 들고 너희 집 찾아간다",
  "다시는 두 발로 못 걷게 해주마", "네 팔다리를 성한 데 없게 만든다", "죽여버리기 전에 당장 꺼져",
  "너 하나 없애는 건 일도 아니야", "내 앞에서 다시 얼쩡대면 끝장이야",
  "잘 생각해라 무슨 일이 생길지 모르니까", "네 가족들 안전 생각하면 조용히 있어",
  "이대로 넘어갈 거라 기대하지는 마라", "언제 어디서 마주칠지 모르니 긴장해",
  "지금부터 매일매일 뒤를 조심해야 할 거야", "네가 아끼는 것들이 무사할 것 같아",
  "곧 아주 나쁜 일이 너한테 닥칠 거다", "웃고 있을 시간도 이제 얼마 안 남았어",
  "I never forget a face, keep that in mind", "we will meet again, count on it",
  "you really brought this on yourself", "better sleep with one eye open now",
  "karma has a way of catching up to people", "your turn is coming sooner than you think",
  "nice little family you have over there", "things can go wrong so very easily",
  "I will end you if you try that again", "I am going to break every bone you have",
  "you are a dead man the moment you show up", "keep talking and you won't walk home",
  "I know exactly where you live", "watch your back every single day from now",
  "you will regret ever crossing me", "I'll make your life a living hell",
  "accidents happen to people who talk too much", "hope your affairs are in order",
  "there won't be a warning next time", "I've hurt people for a lot less than this",
 ],
 "sexual": [
  "오늘 밤 재워줄 사람 필요하지 않아", "둘만 아는 비밀 하나 만들어 볼까", "불 끄면 더 솔직해질 수 있잖아",
  "가까이서 보면 훨씬 더 예쁠 것 같은데", "밤이 좀 더 길었으면 좋겠다 너랑", "네 방 구경 한번 해도 될까",
  "우리 사이에 옷은 좀 거추장스럽지 않아", "지금 너 혼자야 그럼 내가 갈까", "나랑 지금 자고 가지 않을래",
  "옷 다 벗고 침대에서 기다려", "야한 사진 한 장만 보내줘", "네 몸 구석구석 만지고 싶어",
  "가슴 좀 만지게 해줘 지금", "성관계 하고 싶어 참을 수가 없어", "야한 영상 같이 볼래 우리끼리",
  "속옷만 입은 사진 찍어서 보내", "오늘 밤 진하게 한번 보내자", "네 몸매 실물로 보고 싶어 죽겠어",
  "the night is still young and my place is close", "let's make memories we won't be sharing",
  "I keep imagining you without anything on", "come a little closer, nobody has to know",
  "why keep all those clothes on around me", "stay over and I promise we won't sleep",
  "send me a nude photo right this second", "I want to feel every inch of your body",
  "let's have sex tonight, just the two of us", "take everything off for me slowly",
  "I can't stop thinking about your body", "strip down and show me on camera",
  "come to my place and let's get physical", "text me a picture with nothing on",
  "I want you in my bed tonight", "let me touch you all over right now",
 ],
 "gambling": [
  "이번엔 느낌이 확 와 크게 질러야겠어", "잃은 거 이 판에서 한 번에 되찾는다", "판이 커질수록 손맛이 짜릿하지",
  "오늘 안에 본전은 무조건 뽑고 만다", "마지막으로 딱 한 판만 더 돌리자", "이 정도 배짱은 있어야 크게 따지",
  "돈 복사기 돌리러 가는 거지 뭐", "떨어질 때 두 배로 걸어서 만회하면 돼", "카지노 가서 오늘 크게 걸어보자",
  "슬롯머신에 이번 달 월급 다 넣었어", "포커 판돈 더 올려 콜 받는다", "잭팟 노리고 전 재산 걸었어",
  "룰렛에 오늘 느낌대로 다 건다", "이 한 방이면 인생 역전이야 다 걸어", "판돈 두 배로 올려 끝까지 간다",
  "블랙잭에서 오늘 다 딴다 두고 봐", "바카라 자리 잡았으니 크게 지른다", "빚내서라도 이번 판은 들어간다",
  "feeling lucky tonight, time to push it all in", "one more spin and I break even for sure",
  "the bigger the risk the sweeter the payout", "let's chase these losses down tonight",
  "the house always pays out eventually", "go big, fortune favors the bold gambler",
  "put every chip I have on red", "double down, I can feel the win coming",
  "let's hit the casino and bet the rent", "one big hand and we're set for life",
  "all in on this, I'm not backing down", "spin the wheel, max bet, let's go",
  "I bet my whole paycheck on this game", "double or nothing, deal me in",
  "the jackpot is calling my name tonight", "raise the stakes, I'm all in",
 ],
 "drug": [
  "기분 째지는 거 하나 있는데 같이 콜?", "이거 딱 한 입이면 완전 딴 세상이야", "밤샐 각 제대로 나오는 물건 있어",
  "요즘 제일 잘 나가는 걸로 하나 챙겼지", "약국엔 절대 없는 걸로 구해다 줄게", "하늘을 나는 기분 한번 느껴볼래",
  "구름 위를 걷는 느낌이라니까 진짜", "고민 싹 녹여주는 거 내가 쏜다", "필로폰 어디서 구하는지 알아",
  "대마초 좀 말아서 같이 피우자", "코카인 오늘 밤에 한 줄 하자", "약 한 판 제대로 땡기자",
  "히로뽕 좀 구할 데 있으면 알려줘", "주사기 챙겨서 한 대 맞자", "좋은 물건 새로 들어왔는데 해볼래",
  "그거 한 대 빨면 세상이 달라 보여", "이거 하면 밤새 안 자도 끄떡없어", "특별한 걸로 준비했으니까 와",
  "필로폰 좀 구해다 줘 급해", "코카인 파는 데 있으면 연결해줘", "대마 좀 구할 수 있는지 알아봐",
  "약 좀 급하게 구해줘 돈은 준다", "히로뽕 살 수 있는 곳 좀 알려줘", "펜타닐 어디서 구하는지 아는 사람",
  "필로폰 팔 사람 소개 좀 해줘", "코카인 한 봉지만 사게 해줘", "엑스터시 좀 구해줄 수 있어",
  "get me some meth, I can pay well", "where do I buy cocaine around here",
  "hook me up with some pills tonight", "who's selling weed in this area",
  "got something that melts all your worries away", "one little bump and you'll feel golden",
  "this stuff will take you somewhere else", "the good candy just came in fresh",
  "wanna know what floating really feels like", "party favors are on me tonight",
  "where can I score some cocaine tonight", "let's do a line right now",
  "I need to shoot up, got a clean needle", "he's dealing pills on the corner",
  "let's smoke some weed and get high", "one taste of this and you're hooked",
  "buy some meth off me, best price around", "this'll keep you flying all night",
  "fresh batch just dropped, want a hit", "come get high with me tonight",
 ],
}
# safe clauses (added for balance; real negatives dominate at train time)
SAFE = [
  "이번 주말엔 도서관에서 조용히 책이나 볼까", "새로 산 화분에 물 주는 걸 자꾸 까먹네", "다음 달 워크숍 장소를 알아봐야 해",
  "김장 담그는 거 올해도 도와드리기로 했어", "버스 놓쳐서 다음 거 기다리는 중이야", "요즘 아침마다 스트레칭하니까 개운해",
  "발표 끝나고 다 같이 회식하기로 했어요", "택배가 왜 이렇게 안 오는지 모르겠네", "고양이가 자꾸 키보드 위에 올라와",
  "이 매운맛 진짜 죽여준다 또 시켜먹자", "막판 역전골에 심장 떨어지는 줄 알았네", "이번 프로젝트 대박 나서 잭팟 터졌으면",
  "인생 걸고 이 시험 하나에 올인했다", "마약은 한 번이라도 손대면 인생 끝이야", "도박 중독은 정말 무서운 병이더라",
  "필로폰 같은 건 절대 근처도 가면 안 돼", "학교에서 약물 오남용 예방 캠페인을 했어", "폭력은 어떤 이유로도 정당화 못 해",
  "let's plan the team lunch for next friday", "the printer on the third floor is jammed again",
  "I finally finished reading that long novel", "that comedian absolutely killed it last night",
  "we hit the jackpot finding this apartment", "I'm all in on this new hobby of mine",
  "please stay away from drugs, they ruin everything", "gambling wrecked my cousin's whole life",
  "we teach kids to just say no to drugs", "violence never solves a single thing",
  # 죽-idioms — extremely common benign Korean; must NOT read as threats
  "배고파 죽겠다 밥 먹으러 가자", "졸려 죽겠어 좀 자야겠다", "보고 싶어 죽겠어 빨리 와",
  "더워 죽겠다 에어컨 좀 켜자", "웃겨 죽는 줄 알았잖아 진짜", "심심해 죽겠네 뭐 하고 놀지",
  "귀여워 죽겠다 우리 강아지", "좋아 죽겠어 완전 신난다", "이 노래 진짜 죽인다 계속 듣게 돼",
  "그 배우 연기 완전 죽여준다", "여기 경치 죽인다 사진 찍자", "이 집 맛 진짜 죽여줘",
  "분위기 죽인다 데이트하기 딱이야", "실력이 아주 죽여주네 대단하다", "피곤해 죽겠다 얼른 쉬고 싶어",
  "부끄러워 죽겠어 그만 놀려", "긴장돼서 죽는 줄 알았네", "심장 떨어지는 줄 알았잖아",
  "아까워 죽겠다 조금만 더 하지", "예뻐 죽겠다 진짜 잘 어울려",
  "보고 싶어 죽겠어 언제 만나", "그리워 죽겠다 목소리라도 듣고 싶어", "행복해 죽겠어 꿈만 같아",
  "설레 죽겠어 내일이 기다려져", "웃겨 죽겠어 배가 다 아파", "좋아 죽겠다 계속 웃음이 나",
]
CONNECT = {"ko": [" 그리고 ", " 진짜 ", " 야 ", ", "], "en": [" and ", ", ", " — ", " hey, "]}


def _lang(s: str) -> str:
    return "ko" if any("가" <= ch <= "힣" for ch in s) else "en"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=1800)
    args = ap.parse_args()
    rng = random.Random(SEED)

    eval_texts = set()
    for f in ["configs/text/harm_semantic_eval.jsonl", "configs/text/harm_language_testset.jsonl"]:
        for line in (_ROOT / f).read_text().splitlines():
            if line.strip():
                eval_texts.add(json.loads(line)["text"].replace(" ", ""))

    rows, seen = [], set()

    def emit(text, label):
        key = text.replace(" ", "")
        if key in eval_texts or key in seen:
            return
        seen.add(key)
        rows.append({"text": text, "lang": _lang(text), "label": label})

    for cat, base in {**BANK, "safe": SAFE}.items():
        target = args.per_category if cat != "safe" else max(900, args.per_category // 2)
        for s in base:
            emit(s, cat)
        tries = 0
        while sum(1 for r in rows if r["label"] == cat) < target and tries < target * 40:
            tries += 1
            a, b = rng.sample(base, 2)
            if _lang(a) != _lang(b):          # keep each combined example monolingual
                continue
            conn = rng.choice(CONNECT[_lang(a)])
            emit(f"{a}{conn}{b}", cat)

    rng.shuffle(rows)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT.relative_to(_ROOT)}")
    print("by label:", dict(Counter(r["label"] for r in rows)))
    print("by lang :", dict(Counter(r["lang"] for r in rows)))


if __name__ == "__main__":
    main()
