"""Khôi phục dấu tiếng Việt bằng mô hình n-gram âm tiết + giải mã Viterbi.

Ý tưởng: mỗi âm tiết bỏ dấu ('bi') có một tập dạng có dấu ứng viên ('bị', 'bí', 'bì', 'bỉ'...).
Học tần suất unigram/bigram của âm tiết có dấu từ chính kho tài liệu y tế, rồi chọn chuỗi dạng có dấu
có xác suất cao nhất (Viterbi, bigram nội suy với unigram). Không cần GPU hay LLM, chạy vài mili giây.
Âm tiết người dùng đã gõ dấu được giữ nguyên, nên câu gõ lẫn có dấu / không dấu vẫn xử lý được.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .text import fold, has_diacritics, nfc

_TOKEN = re.compile(r"\w+|[^\w]+", re.UNICODE)
_WORD = re.compile(r"\w+", re.UNICODE)


class DiacriticRestorer:
    def __init__(self, max_candidates: int = 8, lam: float = 0.85):
        self.max_candidates = max_candidates
        self.lam = lam
        self.uni: Counter = Counter()
        self.bi: Counter = Counter()
        self.cands: dict[str, list[str]] = {}
        self.total = 0

    # ------------------------------------------------------------------ học
    def fit(self, texts) -> "DiacriticRestorer":
        by_fold: dict[str, Counter] = defaultdict(Counter)
        for text in texts:
            for line in re.split(r"[\n.!?;:]+", nfc(text).lower()):
                syl = _WORD.findall(line)
                if not syl:
                    continue
                self.uni.update(syl)
                self.bi.update(zip(syl, syl[1:]))
                for s in syl:
                    by_fold[fold(s)][s] += 1
        self.total = sum(self.uni.values())
        self.cands = {f: [w for w, _ in c.most_common(self.max_candidates)] for f, c in by_fold.items()}
        return self

    # ------------------------------------------------------------------ xác suất
    def _p_uni(self, w: str) -> float:
        return (self.uni[w] + 1) / (self.total + len(self.uni) + 1)

    def _logp(self, prev: str | None, w: str) -> float:
        pu = self._p_uni(w)
        if prev is None or self.uni[prev] == 0:
            return math.log(pu)
        pb = self.bi[(prev, w)] / self.uni[prev]
        return math.log(self.lam * pb + (1 - self.lam) * pu)

    def _candidates(self, tok: str) -> list[str]:
        low = tok.lower()
        if has_diacritics(low):
            return [low]                       # người dùng đã gõ dấu → giữ nguyên
        return self.cands.get(fold(low), [low])

    # ------------------------------------------------------------------ giải mã
    def restore_words(self, words: list[str]) -> list[str]:
        if not words:
            return []
        lattice = [self._candidates(w) for w in words]
        best = [{c: (self._logp(None, c), None) for c in lattice[0]}]
        for i in range(1, len(words)):
            layer = {}
            for c in lattice[i]:
                score, back = max(((s + self._logp(p, c), p) for p, (s, _) in best[-1].items()),
                                  key=lambda x: x[0])
                layer[c] = (score, back)
            best.append(layer)
        cur = max(best[-1], key=lambda c: best[-1][c][0])
        out = [cur]
        for i in range(len(words) - 1, 0, -1):
            cur = best[i][cur][1]
            out.append(cur)
        return out[::-1]

    def restore(self, text: str) -> str:
        """Trả về câu đã khôi phục dấu, giữ nguyên dấu câu, khoảng trắng và kiểu chữ hoa."""
        tokens = _TOKEN.findall(nfc(text))
        word_pos = [i for i, t in enumerate(tokens) if _WORD.fullmatch(t)]
        restored = self.restore_words([tokens[i] for i in word_pos])
        for i, r in zip(word_pos, restored):
            orig = tokens[i]
            if orig.isupper() and len(orig) > 1:
                r = r.upper()
            elif orig[:1].isupper():
                r = r[:1].upper() + r[1:]
            tokens[i] = r
        return "".join(tokens)


def needs_restoration(text: str, min_words: int = 2) -> bool:
    """Câu có ít nhất `min_words` âm tiết chữ và không có ký tự nào mang dấu tiếng Việt."""
    words = [w for w in _WORD.findall(text) if not w.isdigit()]
    return len(words) >= min_words and not has_diacritics(text)


def syllable_accuracy(restorer: DiacriticRestorer, sentences: list[str]) -> dict:
    """Bỏ dấu từng câu rồi khôi phục, so với câu gốc (theo âm tiết, không phân biệt hoa/thường)."""
    ok = total = exact = 0
    for s in sentences:
        gold = _WORD.findall(nfc(s).lower())
        pred = _WORD.findall(restorer.restore(fold(s)).lower())
        if len(gold) != len(pred):
            total += len(gold)
            continue
        hit = sum(g == p for g, p in zip(gold, pred))
        ok += hit
        total += len(gold)
        exact += hit == len(gold)
    return {"syllable_acc": ok / max(total, 1), "sentence_exact": exact / max(len(sentences), 1),
            "n_sentences": len(sentences), "n_syllables": total}
