"""Sinh trang GitHub Pages (docs/index.html) từ reports/metrics.json + reports/examples.json.

Trang tĩnh, không cần server: phần "Thử như giám khảo" hiển thị kết quả thật của pipeline đã tính sẵn,
nên ai cũng xem được demo kể cả khi Hugging Face Space đang ngủ.
"""
import html
import json
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GITHUB = "https://github.com/chinguyenkhac/ViCare"
COLAB = "https://colab.research.google.com/github/chinguyenkhac/ViCare/blob/main/notebooks/ViCare_Safe_v3.ipynb"


def pct(x, digits=0):
    return "–" if x is None else f"{x * 100:.{digits}f}%"


def build():
    m = json.loads((ROOT / "reports/metrics.json").read_text())
    ex = json.loads((ROOT / "reports/examples.json").read_text())
    r = m["retrieval"]

    # ---------------- ví dụ (markdown → html)
    for e in ex:
        e["html"] = markdown.markdown(html.escape(e["markdown"], quote=False), extensions=["sane_lists"])
    groups, order = {}, []
    for e in ex:
        if e["group"] not in groups:
            groups[e["group"]] = {"title": e["title"], "items": []}
            order.append(e["group"])
        groups[e["group"]]["items"].append({k: e[k] for k in ("variant", "question", "history", "status", "level",
                                                               "prob", "emergency", "understood", "html", "seconds")})
    data = json.dumps({"order": order, "groups": groups}, ensure_ascii=False).replace("</", "<\\/")

    # ---------------- bảng kết quả
    conds = [("Có dấu", "dense|có dấu", "v3|có dấu"),
             ("Không dấu", "dense|không dấu – thô", "v3|không dấu – khôi phục"),
             ("Teencode", "dense|teencode – thô", "v3|teencode – chuẩn hóa")]
    ret_rows = "".join(
        f"<tr><th scope='row'>{n}</th>" + "".join(
            f"<td><span class='old'>{pct(r[b][k])}</span> → <strong>{pct(r[v][k])}</strong></td>"
            for k in ("R@1", "R@5", "R@10", "MRR@10")) + "</tr>"
        for n, b, v in conds)
    rel_rows = "".join(
        f"<tr><th scope='row'><span class='lvl lvl-{i}'>{x['level']}</span></th><td>{x['n']:.0f}</td>"
        f"<td>{pct(x['mean_conf'])}</td><td><strong>{pct(x['actual_acc'])}</strong></td></tr>"
        for i, x in enumerate(m["reliability"]))
    dec_cols = ["Cao", "Trung bình", "Thấp", "Từ chối"]
    dec_rows = "".join(
        f"<tr><th scope='row'>{html.escape(k)}</th>" + "".join(f"<td>{pct(v.get(c, 0))}</td>" for c in dec_cols) + "</tr>"
        for k, v in m["decisions"].items())
    s = m["safety"]
    saf_rows = "".join(
        f"<tr><th scope='row'>{k}</th><td>{pct(s['baseline BTC'].get(k))}</td><td><strong>{pct(s['ViCare-Safe'].get(k))}</strong></td></tr>"
        for k in ("cấp cứu – có dấu", "cấp cứu – không dấu", "báo động giả"))
    saf_rows += f"<tr><th scope='row'>chặn hỏi liều/kê đơn</th><td>–</td><td><strong>{pct(s['ViCare-Safe'].get('chặn hỏi liều/kê đơn'))}</strong></td></tr>"

    nd_b, nd_v = r["dense|không dấu – thô"]["R@5"], r["v3|không dấu – khôi phục"]["R@5"]
    high = next((x for x in m["reliability"] if x["level"] == "Cao"), None)
    stats = [
        (f"{pct(nd_b)} → {pct(nd_v)}", "Recall@5 khi gõ <strong>không dấu</strong>"),
        (pct(m["restoration"]["syllable_acc"], 1), "âm tiết được <strong>khôi phục dấu</strong> đúng"),
        (f"{pct(s['baseline BTC']['cấp cứu – không dấu'])} → {pct(s['ViCare-Safe']['cấp cứu – không dấu'])}",
         "nhận ra <strong>cấp cứu</strong> khi gõ không dấu"),
        (pct(high["actual_acc"]) if high else "–", "tìm đúng tài liệu khi ViCare báo độ tin cậy <strong>Cao</strong>"),
    ]
    stat_html = "".join(f"<div class='stat'><div class='num'>{a}</div><div class='lbl'>{b}</div></div>" for a, b in stats)

    page = TEMPLATE
    for key, val in {"__DATA__": data, "__RET__": ret_rows, "__REL__": rel_rows, "__DEC__": dec_rows,
                     "__SAF__": saf_rows, "__STATS__": stat_html, "__GITHUB__": GITHUB, "__COLAB__": COLAB,
                     "__ECE__": f"{m['calibration']['ECE']:.3f}", "__AUROC__": f"{m['calibration']['AUROC']:.2f}",
                     "__TAU__": f"{m['tau']:.2f}", "__NCAND__": str(m["config"]["N_CAND"])}.items():
        page = page.replace(key, val)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("")
    report = ROOT / "reports/ViCare_Safe_v3.html"
    if report.exists():
        shutil.copy2(report, DOCS / "report.html")
    if not (DOCS / "space.json").exists():
        (DOCS / "space.json").write_text(json.dumps({"url": None}))
    print("✅ docs/index.html")


TEMPLATE = r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ViCare-Safe</title>
<meta name="description" content="Tra cứu sức khỏe tiếng Việt: hiểu câu không dấu và teencode, có độ tin cậy, bằng chứng và nguồn.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#f7f6f3; --surface:#ffffff; --surface-2:#f0eee9; --text:#1d1b18; --muted:#5f5a52; --line:#e2ded6;
  --accent:#c2412d; --accent-ink:#ffffff; --accent-soft:#f8e3dd;
  --high:#1f7a4d; --high-soft:#dcf1e5; --mid:#9a6a00; --mid-soft:#f7ecd0; --low:#b42318; --low-soft:#fbe0dd;
  --shadow:0 1px 2px rgba(0,0,0,.05),0 4px 16px rgba(0,0,0,.05); --radius:14px;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#141311; --surface:#1d1b19; --surface-2:#26231f; --text:#f1eee8; --muted:#aaa397; --line:#34302a;
    --accent:#f0705a; --accent-ink:#1a0d0a; --accent-soft:#3a1f19;
    --high:#5fd497; --high-soft:#16301f; --mid:#f0c45a; --mid-soft:#352a10; --low:#ff8a7a; --low-soft:#3a1916;
    --shadow:none;
  }
}
:root[data-theme="dark"]{
  --bg:#141311; --surface:#1d1b19; --surface-2:#26231f; --text:#f1eee8; --muted:#aaa397; --line:#34302a;
  --accent:#f0705a; --accent-ink:#1a0d0a; --accent-soft:#3a1f19;
  --high:#5fd497; --high-soft:#16301f; --mid:#f0c45a; --mid-soft:#352a10; --low:#ff8a7a; --low-soft:#3a1916;
  --shadow:none;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 "Be Vietnam Pro",system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--accent)}
.wrap{max-width:1080px;margin:0 auto;padding:0 16px}
nav{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
nav .wrap{display:flex;align-items:center;gap:20px;height:56px}
nav .brand{font-weight:700;text-decoration:none;color:var(--text);margin-right:auto}
nav a.l{color:var(--muted);text-decoration:none;font-size:15px}
nav a.l:hover{color:var(--text)}
header.hero{padding:56px 0 24px}
.eyebrow{display:inline-block;font-size:13px;font-weight:600;letter-spacing:.02em;color:var(--accent);background:var(--accent-soft);padding:4px 10px;border-radius:999px}
h1{font-size:clamp(32px,5vw,52px);line-height:1.1;margin:16px 0 12px;letter-spacing:-.02em}
.lead{font-size:clamp(17px,2.2vw,20px);color:var(--muted);max-width:760px;margin:0 0 24px}
.btns{display:flex;flex-wrap:wrap;gap:10px}
.btn{display:inline-flex;align-items:center;gap:8px;padding:10px 16px;border-radius:10px;font-weight:600;text-decoration:none;border:1px solid var(--line);color:var(--text);background:var(--surface)}
.btn.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.btn[aria-disabled="true"]{opacity:.55;pointer-events:none}
.note{margin-top:18px;font-size:14px;color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:32px 0 8px}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:16px;box-shadow:var(--shadow)}
.stat .num{font-size:26px;font-weight:700;letter-spacing:-.01em}
.stat .lbl{color:var(--muted);font-size:14px;margin-top:4px}
section{padding:40px 0}
h2{font-size:clamp(24px,3vw,30px);margin:0 0 8px;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 20px;max-width:780px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}
.explorer{display:grid;grid-template-columns:260px 1fr;overflow:hidden}
.topics{border-right:1px solid var(--line);padding:12px;display:flex;flex-direction:column;gap:6px;background:var(--surface-2)}
.topics button{all:unset;cursor:pointer;padding:10px 12px;border-radius:10px;font-weight:500;color:var(--muted)}
.topics button:hover{background:var(--surface);color:var(--text)}
.topics button[aria-selected="true"]{background:var(--surface);color:var(--text);box-shadow:inset 3px 0 0 var(--accent)}
.topics .sep{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);padding:10px 12px 2px}
.pane{padding:20px;min-width:0}
.variants{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
.variants button{all:unset;cursor:pointer;padding:6px 12px;border:1px solid var(--line);border-radius:999px;font-size:14px;color:var(--muted)}
.variants button[aria-pressed="true"]{background:var(--text);color:var(--bg);border-color:var(--text)}
.q{font-size:18px;font-weight:600;margin:0 0 4px;word-break:break-word}
.hist{font-size:14px;color:var(--muted);margin:0 0 4px}
.meta{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0 14px}
.pill{font-size:13px;font-weight:600;padding:3px 10px;border-radius:999px;background:var(--surface-2);color:var(--muted)}
.lvl{font-size:13px;font-weight:700;padding:3px 10px;border-radius:999px}
.lvl-0{background:var(--high-soft);color:var(--high)} .lvl-1{background:var(--mid-soft);color:var(--mid)} .lvl-2{background:var(--low-soft);color:var(--low)}
.lvl-x{background:var(--surface-2);color:var(--muted)}
.out{border-top:1px solid var(--line);padding-top:14px;overflow-wrap:anywhere}
.out p{margin:0 0 12px} .out ul{margin:0 0 12px;padding-left:20px} .out li{margin:4px 0}
.out sub{color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:15px}
th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
thead th{font-size:13px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
td .old{color:var(--muted)}
.tables{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.tables .card{padding:8px 4px 4px;overflow-x:auto}
.tables h3{font-size:17px;margin:10px 12px 4px}
.tables p{font-size:14px;color:var(--muted);margin:0 12px 8px}
.wide{grid-column:1/-1}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;counter-reset:s}
.step{padding:16px}
.step b{display:block;margin-bottom:4px}
.step b::before{counter-increment:s;content:counter(s) ". ";color:var(--accent)}
.step span{color:var(--muted);font-size:15px}
.limits li{margin:6px 0}
footer{border-top:1px solid var(--line);padding:28px 0 48px;color:var(--muted);font-size:14px}
@media (max-width:860px){
  .stats{grid-template-columns:repeat(2,1fr)}
  .explorer{grid-template-columns:1fr}
  .topics{flex-direction:row;overflow-x:auto;border-right:0;border-bottom:1px solid var(--line)}
  .topics .sep{display:none}
  .topics button{white-space:nowrap}
  .tables,.steps{grid-template-columns:1fr}
  nav a.l.hide-m{display:none}
}
</style>
</head>
<body>
<nav><div class="wrap">
  <a class="brand" href="#">🩺 ViCare-Safe</a>
  <a class="l" href="#thu">Thử demo</a>
  <a class="l" href="#ketqua">Kết quả</a>
  <a class="l hide-m" href="#cachlam">Cách hoạt động</a>
  <a class="l" href="__GITHUB__">GitHub</a>
</div></nav>

<header class="hero"><div class="wrap">
  <span class="eyebrow">AI Challenge 24h · AI4LIFE – ĐH Bách Khoa Hà Nội & Rikkei</span>
  <h1>Tra cứu sức khỏe tiếng Việt,<br>hiểu cả khi bạn gõ không dấu.</h1>
  <p class="lead">ViCare-Safe là trợ lý <strong>hỗ trợ quyết định</strong>: trích nguyên văn lời bác sĩ trong kho tài liệu, kèm
  <strong>độ tin cậy</strong>, <strong>bằng chứng</strong> và <strong>nguồn</strong>. Không chẩn đoán, không kê đơn – và biết nói
  “tôi chưa chắc” khi chưa đủ căn cứ.</p>
  <div class="btns">
    <a class="btn primary" id="live" href="#thu">Thử demo bên dưới</a>
    <a class="btn" href="__COLAB__">Mở notebook trên Colab</a>
    <a class="btn" href="report.html">Báo cáo đánh giá (HTML)</a>
    <a class="btn" href="__GITHUB__">Mã nguồn</a>
  </div>
  <p class="note">⚠️ Demo học thuật. Không thay thế tư vấn của nhân viên y tế. Cấp cứu: gọi <strong>115</strong>.</p>
  <div class="stats">__STATS__</div>
</div></header>

<section id="thu"><div class="wrap">
  <h2>Thử như giám khảo</h2>
  <p class="sub">Chọn một câu hỏi, rồi đổi cách gõ: có dấu, không dấu, teencode. Kết quả dưới đây là đầu ra <em>thật</em> của
  pipeline (chế độ trích xuất, không LLM), được tính sẵn để trang chạy được ở mọi nơi.</p>
  <div class="card explorer">
    <div class="topics" role="tablist" id="topics"></div>
    <div class="pane">
      <div class="variants" id="variants"></div>
      <p class="hist" id="hist" hidden></p>
      <p class="q" id="q"></p>
      <div class="meta" id="meta"></div>
      <div class="out" id="out"></div>
    </div>
  </div>
</div></section>

<section id="ketqua"><div class="wrap">
  <h2>Kết quả đo được</h2>
  <p class="sub">Cùng kho tài liệu và cùng 100 câu dev của notebook BTC. Baseline = notebook gốc (tìm kiếm vector e5).
  Cấu hình giống demo trên CPU: reranker chấm __NCAND__ ứng viên. Chi tiết, khoảng tin cậy 95% và kiểm định McNemar
  trong <a href="report.html">báo cáo</a>.</p>
  <div class="tables">
    <div class="card wide">
      <h3>Truy xuất tài liệu đúng: baseline → ViCare-Safe</h3>
      <table><thead><tr><th>Cách gõ</th><th>Recall@1</th><th>Recall@5</th><th>Recall@10</th><th>MRR@10</th></tr></thead>
      <tbody>__RET__</tbody></table>
    </div>
    <div class="card">
      <h3>Độ tin cậy có nói thật không?</h3>
      <p>Tỷ lệ tài liệu gốc nằm trong 3 nguồn hiển thị, theo mức ViCare tự báo. ECE = __ECE__ · AUROC = __AUROC__.</p>
      <table><thead><tr><th>Mức</th><th>Số câu</th><th>Dự đoán TB</th><th>Thực tế</th></tr></thead><tbody>__REL__</tbody></table>
    </div>
    <div class="card">
      <h3>An toàn</h3>
      <p>Bộ câu tự viết (20 cấp cứu, 15 câu thường, 10 câu hỏi liều) – kết quả có thể lạc quan.</p>
      <table><thead><tr><th></th><th>Baseline</th><th>ViCare-Safe</th></tr></thead><tbody>__SAF__</tbody></table>
    </div>
    <div class="card wide">
      <h3>ViCare quyết định thế nào?</h3>
      <p>Câu ngoài phạm vi bị từ chối khi điểm reranker &lt; τ = __TAU__; còn lại chia mức Cao / Trung bình / Thấp (Thấp = không trả lời, chỉ hiện nguồn).</p>
      <table><thead><tr><th>Bộ câu hỏi</th><th>Cao</th><th>Trung bình</th><th>Thấp</th><th>Từ chối</th></tr></thead><tbody>__DEC__</tbody></table>
    </div>
  </div>
</div></section>

<section id="cachlam"><div class="wrap">
  <h2>Cách hoạt động</h2>
  <p class="sub">Không cần GPU hay LLM khi chạy demo – mọi bước đều chạy được trên CPU miễn phí.</p>
  <div class="steps">
    <div class="card step"><b>Lớp an toàn</b><span>Nhận diện cấp cứu (có dấu lẫn không dấu) → nhắc gọi 115; chặn câu hỏi xin liều thuốc / kê đơn.</span></div>
    <div class="card step"><b>Chuẩn hóa teencode</b><span>“ko, j, dc, bs, bv…” → dạng chuẩn, có luật tránh nhầm (vitamin K, viêm gan C, ng/mL).</span></div>
    <div class="card step"><b>Khôi phục dấu</b><span>Mô hình n-gram học từ chính kho tài liệu + giải mã Viterbi: “con toi bi sot” → “con tôi bị sốt”, &lt;1 ms.</span></div>
    <div class="card step"><b>Truy xuất lai</b><span>BM25 trên chữ bỏ dấu + vector e5, gộp bằng RRF, rồi cross-encoder bge-reranker-v2-m3 xếp hạng lại.</span></div>
    <div class="card step"><b>Độ tin cậy</b><span>Hồi quy logistic trên 8 tín hiệu truy xuất → xác suất tài liệu đúng nằm trong 3 nguồn hiển thị.</span></div>
    <div class="card step"><b>Bằng chứng & nguồn</b><span>Chấm từng câu trong tài liệu, trích nguyên văn câu liên quan nhất kèm số nguồn [n].</span></div>
  </div>
</div></section>

<section><div class="wrap">
  <h2>Giới hạn cần biết</h2>
  <ul class="limits">
    <li><strong>Độ tin cậy</strong> là độ tin vào việc <em>tìm đúng tài liệu</em>, không phải độ đúng y khoa.</li>
    <li>Dữ liệu ViHealthQA (Vinmec, VnExpress) thu thập khoảng 2019–2022 – có thể đã cũ; chỉ dùng cho nghiên cứu.</li>
    <li>Câu không dấu / teencode trong đánh giá là <strong>tổng hợp</strong> từ câu gốc; bộ câu cấp cứu do nhóm tự viết.</li>
    <li>Phát hiện cấp cứu dựa trên từ khóa – luôn gọi 115 khi nghi ngờ cấp cứu.</li>
  </ul>
</div></section>

<footer><div class="wrap">
  Dữ liệu: Nguyen et al., <em>SPBERTQA: A Two-Stage Question Answering System Based on Sentence Transformers for Medical Texts</em>,
  KSEM 2022 · <a href="https://huggingface.co/datasets/tarudesu/ViHealthQA">tarudesu/ViHealthQA</a> ·
  Mã nguồn MIT: <a href="__GITHUB__">__GITHUB__</a>
</div></footer>

<script>
const DATA = __DATA__;
const SPECIAL = new Set(["oos","inject","dose","emerg","followup"]);
const LV = {"Cao":0,"Trung bình":1,"Thấp":2};
const STATUS = {answered:"Đã trả lời",low_confidence:"Độ tin cậy thấp – chỉ hiện nguồn",refused_oos:"Từ chối – ngoài phạm vi",dose_blocked:"Chặn hỏi liều"};
let cur = {g: DATA.order[0], v: 0};
const $ = id => document.getElementById(id);
function renderTopics(){
  const t = $("topics"); t.innerHTML = "";
  let sepDone = false;
  DATA.order.forEach(g => {
    if (SPECIAL.has(g) && !sepDone){ const s = document.createElement("div"); s.className="sep"; s.textContent="Tình huống đặc biệt"; t.appendChild(s); sepDone = true; }
    const b = document.createElement("button");
    b.setAttribute("role","tab"); b.textContent = DATA.groups[g].title;
    b.setAttribute("aria-selected", g === cur.g);
    b.onclick = () => { cur = {g, v: 0}; render(); };
    t.appendChild(b);
  });
}
function render(){
  renderTopics();
  const grp = DATA.groups[cur.g], it = grp.items[cur.v];
  const vs = $("variants"); vs.innerHTML = "";
  if (grp.items.length > 1) grp.items.forEach((x, i) => {
    const b = document.createElement("button"); b.textContent = x.variant;
    b.setAttribute("aria-pressed", i === cur.v); b.onclick = () => { cur.v = i; render(); }; vs.appendChild(b);
  });
  $("q").textContent = "❓ " + it.question;
  const h = $("hist"); if (it.history){ h.hidden = false; h.textContent = "Lịch sử: " + it.history[it.history.length-1]; } else h.hidden = true;
  const meta = [];
  meta.push(`<span class="pill">${STATUS[it.status] || it.status}</span>`);
  if (it.level) meta.push(`<span class="lvl lvl-${LV[it.level]}">Độ tin cậy: ${it.level} (${Math.round(it.prob*100)}%)</span>`);
  if (it.emergency) meta.push(`<span class="lvl lvl-2">Cảnh báo cấp cứu</span>`);
  meta.push(`<span class="pill">${it.seconds}s trên CPU</span>`);
  $("meta").innerHTML = meta.join("");
  $("out").innerHTML = it.html;
}
render();
fetch("space.json").then(r => r.ok ? r.json() : null).then(j => {
  if (j && j.url){ const a = $("live"); a.href = j.url; a.textContent = "Mở demo trực tiếp (Hugging Face)"; }
}).catch(() => {});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    build()
