"""Phase 1.6 (①) — learned text-harm head: frozen multilingual-e5 + trainable MLP.

Replaces the prototype-cosine decision with a *learned* boundary, to lift implicit-harm
recall (the measured ceiling) and cut real-world false positives. Training data mixes
implicit-heavy synthetic harm (NEW sentences, disjoint from the eval sets) + synthetic
safe + REAL benign Korean (kor_unsmile TRAIN clean) — the real negatives directly attack
the 3.5% real FP. Frozen e5 embeddings are cached, so training the small head is CPU-fast.

Eval is on the held-out sets only (470 synthetic + 41 real-speech transcripts + kor_unsmile
VALID clean), compared against the prototype baseline. No GPU, no LLM.

    uv run python scripts/train_text_head.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

sys.path.insert(0, str(_ROOT / "src"))

CATS = ["threat", "sexual", "gambling", "drug", "abuse", "safe"]
CAT_IDX = {c: i for i, c in enumerate(CATS)}
SAFE_IDX = CAT_IDX["safe"]
MODEL_OUT = _ROOT / "artifacts" / "text_head.pt"
SEED = 1234

# --- implicit-heavy synthetic banks (NEW sentences, kept disjoint from eval) ---
BANK = {
 "threat": [
  "네가 한 짓 절대 안 잊는다", "우리 다시 마주칠 날이 반드시 온다", "내가 누군지 곧 뼈저리게 알게 될 거다",
  "밤길 조심해라 세상 참 험하니까", "그 대가는 반드시 치르게 만들어주지", "너희 식구들도 마음 편치 못할 거다",
  "지금 웃을 수 있는 것도 얼마 안 남았어", "어디 한번 끝까지 가보자 두고 봐", "네 인생 이제부터 꼬이기 시작할 거다",
  "오늘 일 두고두고 후회하게 해주마", "곱게 넘어갈 거라는 착각은 버려", "언젠가 골목에서 딱 마주치겠지",
  "I never forget a face, keep that in mind", "we will meet again, you can count on it",
  "you really brought this on yourself now", "better sleep with one eye open from now",
  "karma has a way of catching up to people", "your turn is coming sooner than you think",
  "nice little family you have over there", "things can go wrong so very easily you know",
 ],
 "sexual": [
  "오늘 밤 재워줄 사람 필요하지 않아", "둘만 아는 비밀 하나 만들어 볼까", "불 끄면 더 솔직해질 수 있잖아",
  "가까이서 보면 훨씬 더 예쁠 것 같은데", "밤이 좀 더 길었으면 좋겠다 너랑", "네 방 구경 한번 해도 될까",
  "우리 사이에 옷은 좀 거추장스럽지 않아", "지금 너 혼자야 그럼 내가 갈까",
  "the night is still young and my place is close", "let's make memories we won't be sharing",
  "I keep imagining you without anything on", "come a little closer, nobody has to know",
  "why keep all those clothes on around me", "stay over and I promise we won't sleep",
 ],
 "gambling": [
  "이번엔 느낌이 확 와 크게 질러야겠어", "잃은 거 이 판에서 한 번에 되찾는다", "판이 커질수록 손맛이 짜릿하지",
  "오늘 안에 본전은 무조건 뽑고 만다", "마지막으로 딱 한 판만 더 돌리자", "이 정도 배짱은 있어야 크게 따지",
  "돈 복사기 돌리러 가는 거지 뭐", "떨어질 때 두 배로 걸어서 만회하면 돼",
  "feeling lucky tonight, time to push it all in", "one more spin and I break even for sure",
  "the bigger the risk the sweeter the payout", "let's chase these losses down tonight",
  "the house always pays out eventually right", "go big, fortune favors the bold gambler",
 ],
 "drug": [
  "기분 째지는 거 하나 있는데 같이 콜?", "이거 딱 한 입이면 완전 딴 세상이야", "밤샐 각 제대로 나오는 물건 있어",
  "요즘 제일 잘 나가는 걸로 하나 챙겼지", "약국엔 절대 없는 걸로 구해다 줄게", "하늘을 나는 기분 한번 느껴볼래",
  "구름 위를 걷는 느낌이라니까 진짜", "고민 싹 녹여주는 거 내가 쏜다",
  "got something that melts all your worries away", "one little bump and you'll feel golden",
  "this stuff will take you somewhere else entirely", "the good candy just came in fresh",
  "wanna know what floating really feels like", "party favors are on me tonight, come by",
 ],
 "safe": [
  "이번 주말엔 도서관에서 조용히 책이나 볼까", "새로 산 화분에 물 주는 걸 자꾸 까먹네", "다음 달 워크숍 장소를 알아봐야 해",
  "김장 담그는 거 올해도 도와드리기로 했어", "버스 놓쳐서 다음 거 기다리는 중이야", "요즘 아침마다 스트레칭하니까 개운해",
  "발표 끝나고 다 같이 회식하기로 했어요", "택배가 왜 이렇게 안 오는지 모르겠네", "고양이가 자꾸 키보드 위에 올라와",
  "주말 등산 갔다가 다리 완전 뻐근하다", "이 드라마 결말이 진짜 궁금해 죽겠어", "그 식당 웨이팅이 너무 길어서 포기했어",
  "let's plan the team lunch for next friday", "the printer on the third floor is jammed again",
  "I finally finished reading that long novel", "the garden needs weeding before it rains",
  "can you send me the slides from yesterday", "my flight got pushed back to the evening",
  # tricky idioms (benign despite harm words)
  "이 매운맛 진짜 죽여준다 또 시켜먹자", "막판 역전골에 심장 떨어지는 줄 알았네", "그 사람 개그 감각은 진짜 미쳤어",
  "이번 프로젝트 대박 나서 잭팟 터졌으면", "인생 걸고 이 시험 하나에 올인했다", "웃겨 죽는 줄 알았잖아 진짜",
  "that comedian absolutely killed it last night", "we hit the jackpot finding this apartment",
  "I'm all in on this new hobby of mine", "that spicy ramen nearly killed me honestly",
  # stance / condemnation (names harm to reject it)
  "마약은 한 번이라도 손대면 인생 끝이야", "도박 중독은 정말 무서운 병이더라", "필로폰 같은 건 절대 근처도 가면 안 돼",
  "학교에서 약물 오남용 예방 캠페인을 했어", "폭력은 어떤 이유로도 정당화 못 해", "친구가 도박 끊고 착실히 살고 있어",
  "please stay away from drugs, they ruin everything", "gambling wrecked my cousin's whole life",
  "we teach kids to just say no to drugs", "violence never solves a single thing",
 ],
}


class Head(nn.Module):
    def __init__(self, d=768, h=256, n=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.3), nn.Linear(h, n))

    def forward(self, x):
        return self.net(x)


def _embed(model, texts, prefix):
    return model.encode([f"{prefix}: {t}" for t in texts], normalize_embeddings=True,
                        batch_size=64, show_progress_bar=False)


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = SentenceTransformer("intfloat/multilingual-e5-base")

    # eval texts to exclude from training (avoid leakage)
    eval_texts = set()
    for f in ["configs/text/harm_semantic_eval.jsonl", "configs/text/harm_language_testset.jsonl"]:
        for line in (_ROOT / f).read_text().splitlines():
            if line.strip():
                eval_texts.add(json.loads(line)["text"].replace(" ", ""))

    # --- build training set: research-scale synthetic corpus + REAL negatives ---
    train_texts, train_labels = [], []
    corpus = _ROOT / "configs/text/harm_train_corpus.jsonl"
    if corpus.exists():
        for line in corpus.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["text"].replace(" ", "") in eval_texts:
                continue
            train_texts.append(r["text"])
            train_labels.append(CAT_IDX[r["label"]])
    else:  # fallback to the small inline banks
        for cat, sents in BANK.items():
            for s in sents:
                if s.replace(" ", "") not in eval_texts:
                    train_texts.append(s)
                    train_labels.append(CAT_IDX[cat])
    # real Korean from kor_unsmile TRAIN: clean -> negatives (attacks FP); 악플/욕설 -> real
    # toxic POSITIVES labeled threat. Synthetic-only positives had ~0% real-world recall;
    # real toxic teaches the actual violent/hostile vocabulary ('칼빵','총살','늑골 아작').
    import re
    vpat = re.compile(r"죽여|죽인|죽어|총살|쏴|쏘|칼|패버|때려|후려|숨통|끝장|없애|족쳐|"
                      r"파묻|린치|목졸|찔러|살인|테러|폭행|구타|주먹|팰|뚝배기|대가리")
    p = hf_hub_download("smilegate-ai/kor_unsmile", "data/train-00000-of-00001.parquet", repo_type="dataset")
    dftr = pd.read_parquet(p)
    clean = dftr[dftr["clean"] == 1]["문장"].tolist()
    # real toxic POSITIVES: 악플/욕설 with a VIOLENCE marker -> threat; the rest -> abuse
    # (profanity/insults are harmful to minors even without a threat, so we keep them —
    # a dedicated 'abuse' class avoids the FP blow-up from mislabeling all profanity as threat).
    # abuse positives = only STRONG explicit slurs (not casual filler profanity, which
    # overlaps normal informal Korean and blew FP up to 31%). Ambiguous mild 악플 is skipped.
    strong = re.compile(r"씨발|시발|병신|ㅄ|ㅂㅅ|개새끼|개새|개색|좆|존나게|지랄|썅|또라이|미친놈|"
                        r"미친새|니미|느금|엄창|창녀|걸레|후장|보지|자지|꺼져라|닥쳐라|엿먹어")
    toxic = [str(s) for s in dftr[dftr["악플/욕설"] == 1]["문장"]]
    rng.shuffle(clean)
    for s in clean[:3000]:
        train_texts.append(str(s))
        train_labels.append(SAFE_IDX)
    n_threat = n_abuse = n_skip = 0
    for s in toxic:
        if vpat.search(s):
            train_labels.append(CAT_IDX["threat"]); train_texts.append(s); n_threat += 1
        elif strong.search(s):
            train_labels.append(CAT_IDX["abuse"]); train_texts.append(s); n_abuse += 1
        else:
            n_skip += 1  # ambiguous casual profanity -> not a training positive
    print(f"real toxic positives: threat(violent) {n_threat}, abuse(strong slur) {n_abuse}, "
          f"skipped(ambiguous) {n_skip}")
    y = np.array(train_labels)
    print(f"train: {len(train_texts)} ({np.bincount(y)}  = {CATS})")

    print("embedding training set (frozen e5)...")
    X = np.asarray(_embed(model, train_texts, "query"), dtype=np.float32)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y).long()

    # class-weighted loss (safe dominates)
    counts = np.bincount(y, minlength=len(CATS)).astype(np.float32)
    w = torch.from_numpy(counts.sum() / (len(CATS) * np.maximum(counts, 1))).float()

    head = Head(n=len(CATS))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)
    n = len(Xt)
    idx = np.arange(n)
    for _ep in range(60):
        head.train()
        rng.shuffle(idx)
        for i in range(0, n, 128):
            b = idx[i:i + 128]
            opt.zero_grad()
            loss = lossf(head(Xt[b]), yt[b])
            loss.backward()
            opt.step()
    torch.save({"state": head.state_dict(), "cats": CATS}, MODEL_OUT)
    print(f"trained head saved -> {MODEL_OUT.relative_to(_ROOT)}")

    # --- eval helper ---
    head.eval()

    @torch.no_grad()
    def risk_of(texts):
        e = torch.from_numpy(np.asarray(_embed(model, texts, "query"), dtype=np.float32))
        prob = torch.softmax(head(e), dim=1).numpy()
        return 1.0 - prob[:, SAFE_IDX]  # harmful risk

    def prf(risks, harmful, thr=0.5):
        risks = np.asarray(risks); harmful = np.asarray(harmful)
        pred = risks >= thr
        tp = int((pred & harmful).sum()); fp = int((pred & ~harmful).sum()); fn = int((~pred & harmful).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return round(p, 3), round(r, 3), round(2 * p * r / (p + r), 3) if p + r else 0.0

    ho_path = _ROOT / "configs/text/harm_language_testset.jsonl"
    rows = [json.loads(x) for x in ho_path.read_text().splitlines() if x.strip()]
    r470 = np.asarray(risk_of([r["text"] for r in rows]))
    h470 = [r["label"] != "safe" for r in rows]
    imp = [i for i, r in enumerate(rows) if r.get("kind") == "implicit"]
    imp_recall = float((r470[imp] >= 0.5).mean())
    # real speech 41 — overall + per language
    rs = json.loads((_ROOT / "artifacts/eval_real_speech.json").read_text())["records"]
    rrs = np.asarray(risk_of([r["asr"] for r in rs]))
    hrs = np.asarray([r["gold"] != "safe" for r in rs])
    ko = np.array([r["lang"] == "ko" for r in rs])
    # EXTERNAL held-out (kor_unsmile VALID — NOT in training):
    #   clean -> false-positive rate;  악플/욕설 with violence markers -> real threat recall
    import re
    pv = hf_hub_download("smilegate-ai/kor_unsmile", "data/valid-00000-of-00001.parquet", repo_type="dataset")
    vdf = pd.read_parquet(pv)
    vpat = re.compile(r"죽여|죽인|죽어|총살|쏴|쏘|칼|패버|때려|후려|숨통|끝장|없애|족쳐|파묻|린치|목졸|찔러|살인|테러")
    prof = re.compile(r"씨발|시발|존나|졸라|병신|ㅄ|ㅂㅅ|새끼|개새|지랄|꺼져|닥쳐|엿먹|좆|년|놈|미친|또라이|썅|개같")
    vclean = [str(t) for t in vdf[vdf["clean"] == 1]["문장"]]
    fp_rate = float((np.asarray(risk_of(vclean)) >= 0.5).mean())
    genuine = [t for t in vclean if not prof.search(t)]        # clean AND no profanity
    fp_genuine = float((np.asarray(risk_of(genuine)) >= 0.5).mean())
    vabuse = [str(t) for t in vdf[vdf["악플/욕설"] == 1]["문장"]]
    ab_recall = float((np.asarray(risk_of(vabuse)) >= 0.5).mean())
    vthreat = [str(t) for t in vdf[vdf["악플/욕설"] == 1]["문장"] if vpat.search(str(t))]
    ext_recall = float((np.asarray(risk_of(vthreat)) >= 0.5).mean()) if vthreat else 0.0

    print("\n=== LEARNED HEAD (e5 + MLP) — with abuse class + real toxic positives ===")
    print(f"held-out 470     P/R/F1 = {prf(r470, h470)}   implicit recall {imp_recall:.3f}")
    print(f"real-speech 41   P/R/F1 = {prf(rrs, hrs)}   "
          f"KO {prf(rrs[ko], hrs[ko])[1]:.2f}/EN {prf(rrs[~ko], hrs[~ko])[1]:.2f}")
    print("--- EXTERNAL (kor_unsmile VALID, held-out) ---")
    print(f"  threat recall (violence 악플, n={len(vthreat)}) = {ext_recall:.1%}   (0% synthetic-only)")
    print(f"  abuse recall  (all 악플/욕설, n={len(vabuse)}) = {ab_recall:.1%}")
    print(f"  FP on clean (raw)          = {fp_rate:.1%}   (note: 6% of 'clean' is profane)")
    print(f"  FP on GENUINE clean (no profanity, n={len(genuine)}) = {fp_genuine:.1%}   [honest FP]")
    print(f"EXTERNAL threat recall (VALID 악플 violence subset, n={len(vthreat)}) = "
          f"{ext_recall:.1%}   (was 0.0% synthetic-only)   [THE key metric]")


if __name__ == "__main__":
    main()
