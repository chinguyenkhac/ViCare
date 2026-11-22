"""Pipeline ViCare-Safe (chế độ trích xuất – không LLM) và định dạng đầu ra hỗ trợ quyết định."""
from __future__ import annotations

from dataclasses import dataclass, field

from .confidence import ConfidenceModel, features, level_of
from .diacritics import DiacriticRestorer, needs_restoration
from .evidence import extract_evidence
from .safety import DOSE_MSG, EMERGENCY_MSG, is_dose_request, is_emergency
from .text import fold, match_any, normalize_teencode

WARNING = "⚠️ *Thông tin chỉ mang tính tham khảo, không thay thế tư vấn của nhân viên y tế. Dữ liệu thu thập khoảng 2019–2022.*"
REFUSAL = "Tôi không tìm thấy thông tin này trong kho tài liệu."
CONF_NOTE = "khả năng tài liệu phù hợp nằm trong các nguồn dưới đây – không phải độ chính xác y khoa"

FOLLOWUP_START = ("vay ", "the con", "the thi", "nhu vay", "neu vay", "con truong hop", "con tre", "con nguoi")
FOLLOWUP_ANY = ["truong hop nay", "truong hop do", "thi sao", "benh nay", "thuoc nay", "cai do", "dieu do",
                "tinh trang nay", "tinh trang do"]


def build_restorer(kb, question_weight: int = 3) -> DiacriticRestorer:
    """Học n-gram từ kho tài liệu + câu hỏi tập train (văn phong gần người dùng, nhân trọng số)."""
    train_q = [str(r["question"]) for r in kb.rows.get("train", [])]
    return DiacriticRestorer().fit([d["text"] for d in kb.corpus] + train_q * question_weight)


def is_followup(question: str) -> bool:
    f = fold(question).strip()
    return len(f.split()) <= 6 or f.startswith(FOLLOWUP_START) or match_any(f, FOLLOWUP_ANY)


@dataclass
class Understood:
    original: str
    query: str
    normalized: bool = False
    restored: bool = False
    followup: bool = False

    @property
    def changed(self) -> bool:
        return self.query.strip() != self.original.strip()


@dataclass
class Answer:
    status: str                    # answered | low_confidence | refused_oos | dose_blocked | empty
    understood: Understood | None
    level: str | None = None
    prob: float | None = None
    emergency: bool = False
    hits: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    markdown: str = ""


class ViCareSafe:
    def __init__(self, retriever, restorer: DiacriticRestorer, conf: ConfidenceModel, tau: float,
                 n_cand: int = 10, n_show: int = 3):
        self.retriever, self.restorer, self.conf = retriever, restorer, conf
        self.tau, self.n_cand, self.n_show = tau, n_cand, n_show

    # ------------------------------------------------------------------ hiểu câu hỏi
    def preprocess(self, text: str) -> tuple[str, bool, bool]:
        q, normalized = normalize_teencode(str(text).strip())
        restored = False
        if needs_restoration(q):
            r = self.restorer.restore(q)
            restored = r != q
            q = r
        return q, normalized, restored

    def understand(self, question: str, history: list[str] | None = None) -> Understood:
        q, normalized, restored = self.preprocess(question)
        followup = False
        if history and is_followup(question):
            prev, _, _ = self.preprocess(history[-1])
            q, followup = f"{prev} {q}", True
        return Understood(str(question).strip(), q, normalized, restored, followup)

    # ------------------------------------------------------------------ hỏi đáp
    def ask(self, question: str, history: list[str] | None = None, with_evidence: bool = True) -> Answer:
        if not str(question or "").strip():
            return Answer("empty", None, markdown="Vui lòng nhập một câu hỏi.")
        u = self.understand(question, history)
        emergency = is_emergency(u.original) or is_emergency(u.query)
        hits = self.retriever.rerank(u.query, k=self.n_cand, n_cand=self.n_cand)
        top = hits[0]["score"] if hits else 0.0
        shown = hits[: self.n_show]

        if is_dose_request(u.original) or is_dose_request(u.query):
            ans = Answer("dose_blocked", u, emergency=emergency, hits=shown if top >= self.tau else [])
        elif top < self.tau:
            ans = Answer("refused_oos", u, emergency=emergency, hits=[])
        else:
            p = self.conf.predict_one(features(hits, u.query, u.restored, u.normalized))
            lvl = level_of(p)
            ev = extract_evidence(u.query, shown, getattr(self.retriever, "reranker", None)) if with_evidence else []
            status = "low_confidence" if lvl == "Thấp" else "answered"
            ans = Answer(status, u, level=lvl, prob=p, emergency=emergency, hits=shown, evidence=ev)
        ans.markdown = format_answer(ans)
        return ans


# ---------------------------------------------------------------------- định dạng
def _sources(hits: list[dict]) -> str:
    return "\n".join(f"[{i}] {h['url'] if str(h['url']).startswith('http') else h['doc_id']}"
                     for i, h in enumerate(hits, 1))


def format_answer(a: Answer) -> str:
    parts = []
    if a.emergency:
        parts.append(EMERGENCY_MSG)
    u = a.understood
    if u and u.changed:
        why = [w for w, f in (("chuẩn hóa viết tắt", u.normalized), ("khôi phục dấu", u.restored),
                              ("ghép với câu hỏi trước", u.followup)) if f]
        parts.append(f"🔎 **ViCare hiểu câu hỏi là:** “{u.query}” *({', '.join(why)})*")

    if a.status == "dose_blocked":
        parts.append(DOSE_MSG)
        if a.hits:
            parts.append("**📚 Tài liệu liên quan để đọc thêm:**\n" + _sources(a.hits))
    elif a.status == "refused_oos":
        parts.append(f"{REFUSAL}\n\n**➡️ Gợi ý:** câu hỏi có vẻ nằm ngoài phạm vi kho tài liệu sức khỏe; "
                     "hãy hỏi trực tiếp bác sĩ hoặc cơ sở y tế.")
    elif a.status in ("answered", "low_confidence"):
        badge = {"Cao": "🟢", "Trung bình": "🟡", "Thấp": "🔴"}[a.level]
        parts.append(f"**📊 Độ tin cậy: {badge} {a.level} ({a.prob:.0%})** – *{CONF_NOTE}*")
        quotes = "\n".join(f"- «{e['sentence']}» [{e['source']}]" for e in a.evidence)
        if a.status == "answered":
            parts.append("**📌 Thông tin tham khảo** *(trích nguyên văn từ tài liệu bác sĩ)*:\n" + (quotes or "- (xem nguồn)"))
            nxt = ("dùng thông tin này để chuẩn bị câu hỏi khi đi khám; không tự ý bắt đầu, ngừng hoặc đổi thuốc."
                   if a.level == "Cao" else
                   "độ tin cậy chưa cao – hãy đối chiếu các nguồn bên dưới và hỏi thêm nhân viên y tế.")
        else:
            parts.append("**🔴 Độ tin cậy THẤP – ViCare không đưa ra câu trả lời.** "
                         "Các đoạn dưới đây *có thể* liên quan, hãy tự đọc kỹ nguồn:\n" + (quotes or "- (xem nguồn)"))
            nxt = "hỏi trực tiếp bác sĩ; có thể diễn đạt lại câu hỏi cụ thể hơn (triệu chứng, thời gian, tuổi)."
        parts.append(f"**➡️ Bước tiếp theo gợi ý:** {nxt}")
        parts.append("**📚 Nguồn:**\n" + _sources(a.hits))
    parts.append(WARNING)
    return "\n\n".join(parts)
