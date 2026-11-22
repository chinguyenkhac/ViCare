"""Xử lý văn bản tiếng Việt: bỏ dấu, tách âm tiết, tách câu, chuẩn hóa teencode."""
from __future__ import annotations

import re
import unicodedata

VN_DIACRITIC_CHARS = frozenset(
    "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
)

_WORD = re.compile(r"\w+", re.UNICODE)


def fold(text: str) -> str:
    """Bỏ dấu + chữ thường: 'Khó Thở' -> 'kho tho'."""
    s = unicodedata.normalize("NFD", str(text).lower().replace("đ", "d").replace("Đ", "d"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", s)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", str(text))


def has_diacritics(text: str) -> bool:
    return any(c in VN_DIACRITIC_CHARS for c in nfc(text).lower())


def syllables(text: str) -> list[str]:
    """Âm tiết (chữ thường, giữ dấu)."""
    return _WORD.findall(nfc(text).lower())


def match_any(text: str, phrases) -> bool:
    """Có cụm nào xuất hiện trọn vẹn theo ranh giới từ không."""
    return any(re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", text) for p in phrases)


# ----------------------------------------------------------------------------- tách câu
_GREETING = re.compile(
    r"^(xin )?chào|cảm ơn|cám ơn|trân trọng|thân ái|chúc (bạn|anh|chị|em|cháu)|"
    r"(bạn|anh|chị|em) (có thể )?(liên hệ|đặt lịch|gọi)|hotline|tổng đài|đặt lịch",
    re.IGNORECASE,
)


def split_sentences(text: str, min_words: int = 4) -> list[str]:
    """Tách câu, xử lý cả lỗi dính dấu chấm ('...khoa.Em cần...') trong dữ liệu crawl."""
    t = nfc(text).replace("\r", "\n")
    t = re.sub(r"(?<=[a-zà-ỹ0-9\)])\.(?=[A-ZĐÀ-Ỹ])", ". ", t)
    parts = re.split(r"(?<=[.!?…])\s+|\n+", t)
    out = []
    for p in parts:
        p = p.strip(" -•*\t")
        if len(p.split()) >= min_words:
            out.append(p)
    return out


def is_boilerplate(sentence: str) -> bool:
    return bool(_GREETING.search(sentence.strip()[:60]))


# ----------------------------------------------------------------------------- teencode
# Chỉ dùng các viết tắt ít nhập nhằng. Không chuẩn hóa 'a', 'e', 'b', 'c', 'v' vì dễ trùng
# với 'viêm gan A/B/C', 'cúm A', 'vitamin E'...
TEENCODE = {
    "ko": "không", "hok": "không", "hông": "không", "khum": "không", "kh": "không", "hk": "không", "kg": "không", "k": "không",
    "j": "gì", "ji": "gì",
    "dc": "được", "đc": "được", "dk": "được", "đk": "được", "dcj": "được",
    "bs": "bác sĩ", "bsi": "bác sĩ", "bv": "bệnh viện",
    "vs": "với", "z": "vậy", "zậy": "vậy", "dzậy": "vậy", "zay": "vậy",
    "ntn": "như thế nào", "mn": "mọi người",  # không dùng "ng" (trùng đơn vị ng/mL)
    "mk": "mình", "mik": "mình", "mh": "mình",
    "r": "rồi", "cx": "cũng", "cg": "cũng", "lm": "làm", "trc": "trước", "ns": "nói",
    "sk": "sức khỏe", "xn": "xét nghiệm", "tks": "cảm ơn", "thks": "cảm ơn",
}
# Không đổi chữ cái đơn khi đứng sau các từ này ('vitamin K', 'nhóm máu R'...).
_LETTER_GUARD = {"vitamin", "nhóm", "type", "tuýp", "kali", "gan", "cúm", "mũi", "máu", "virus", "vi", "rút"}

_TOKEN = re.compile(r"\w+|[^\w\s]+|\s+", re.UNICODE)
_ELONG = re.compile(r"([^\W\d_])\1{2,}", re.UNICODE)


def normalize_teencode(text: str) -> tuple[str, bool]:
    """Chuẩn hóa teencode. Trả về (câu mới, có thay đổi hay không)."""
    src = _ELONG.sub(r"\1", nfc(text))
    tokens = _TOKEN.findall(src)
    out, prev_word, changed = [], "", src != nfc(text)
    for tok in tokens:
        low = tok.lower()
        single = len(low) == 1
        if low in TEENCODE and not (single and (tok.isupper() or prev_word in _LETTER_GUARD)):
            rep = TEENCODE[low]
            out.append(rep.capitalize() if tok[:1].isupper() and len(tok) > 1 else rep)
            changed = True
        else:
            out.append(tok)
        if tok.strip() and _WORD.fullmatch(tok):
            prev_word = low
    return "".join(out), changed


# Sinh câu teencode tổng hợp (dùng để kiểm thử – dữ liệu gốc gần như không có teencode).
_TO_TEEN = [
    ("như thế nào", "ntn"), ("bác sĩ", "bs"), ("bệnh viện", "bv"), ("xét nghiệm", "xn"),
    ("không", "ko"), ("được", "dc"), ("gì", "j"), ("với", "vs"), ("vậy", "z"),
    ("rồi", "r"), ("cũng", "cx"), ("làm", "lm"), ("trước", "trc"), ("mình", "mk"),
]


def to_teencode(text: str) -> str:
    s = nfc(text)
    for word, short in _TO_TEEN:
        s = re.sub(r"(?<!\w)" + re.escape(word) + r"(?!\w)", short, s, flags=re.IGNORECASE)
    return s


# ----------------------------------------------------------------------------- PII
PII = re.compile(r"(?<!\w)(?:\+?84|0)\d{9,10}(?!\d)|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")


def mask_pii(text: str) -> str:
    return PII.sub("[đã ẩn]", str(text))
