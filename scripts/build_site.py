"""Sinh trang GitHub Pages (docs/index.html) từ reports/metrics.json + reports/examples.json.

Trang tĩnh, không cần server: phần "Thử như giám khảo" hiển thị kết quả thật của pipeline đã tính sẵn,
nên ai cũng xem được demo kể cả khi Hugging Face Space đang ngủ.
"""
import html
import json
import re
import shutil
from pathlib import Path

import markdown

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vicare_safe.confidence import norm_reliability  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GITHUB = "https://github.com/chinguyenkhac/ViCare"
COLAB = "https://colab.research.google.com/github/chinguyenkhac/ViCare/blob/main/notebooks/ViCare_Safe_v3.ipynb"


URL = re.compile(r"(?<![\"'>=])(https?://[^\s<\"']+)")


def linkify(fragment: str) -> str:
    """Gắn thẻ <a> cho URL nguồn (sau khi đã escape nội dung)."""
    return URL.sub(lambda m: f'<a href="{m.group(1)}" rel="noopener" target="_blank">{m.group(1)}</a>', fragment)


def pct(x, digits=0):
    return "–" if x is None else f"{x * 100:.{digits}f}%"


def build(artifact_dir=None):
    m = json.loads((ROOT / "reports/metrics.json").read_text())
    ex = json.loads((ROOT / "reports/examples.json").read_text())
    r = m["retrieval"]
    m["reliability"] = norm_reliability(m["reliability"])

    # ---------------- ví dụ (markdown → html)
    for e in ex:
        e["html"] = linkify(markdown.markdown(html.escape(e["markdown"], quote=False), extensions=["sane_lists", "nl2br"]))
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
             ("Teencode", "dense|teencode – thô", "v3|teencode – chuẩn hóa"),
             ("Teencode + không dấu", "dense|teencode + không dấu – thô", "v3|teencode + không dấu – xử lý")]
    ret_rows = "".join(
        f"<tr><th scope='row'>{n}</th>" + "".join(
            f"<td><span class='old'>{pct(r[b][k])}</span> → <strong>{pct(r[v][k])}</strong></td>"
            for k in ("R@1", "R@5", "R@10", "MRR@10")) + "</tr>"
        for n, b, v in conds)
    rel_rows = "".join(
        f"<tr><th scope='row'><span class='chip t{i}'>{x['level']}</span></th><td>{x['n']:.0f}</td>"
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

    body = BODY
    for key, val in {"__DATA__": data, "__RET__": ret_rows, "__REL__": rel_rows, "__DEC__": dec_rows,
                     "__SAF__": saf_rows, "__STATS__": stat_html, "__GITHUB__": GITHUB, "__COLAB__": COLAB,
                     "__ECE__": f"{m['calibration']['ECE']:.3f}", "__AUROC__": f"{m['calibration']['AUROC']:.2f}",
                     "__TAU__": f"{m['tau']:.2f}", "__NCAND__": str(m["config"]["N_CAND"])}.items():
        body = body.replace(key, val)
    DOCS.mkdir(exist_ok=True)
    page = ('<!doctype html>\n<html lang="vi">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + HEAD + "\n</head>\n<body>\n" + body + "\n" + PAGES_EXTRA + "\n</body>\n</html>\n")
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("")
    report = ROOT / "reports/ViCare_Safe_v3.html"
    if report.exists():
        shutil.copy2(report, DOCS / "report.html")
    if not (DOCS / "space.json").exists():
        (DOCS / "space.json").write_text(json.dumps({"url": None}))
    if artifact_dir:   # bản cho claude.ai Artifact: chỉ phần nội dung (khung HTML do nền tảng thêm)
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "vicare_safe.html").write_text(HEAD + "\n" + body + "\n", encoding="utf-8")
        if report.exists():
            shutil.copy2(report, artifact_dir / "report.html")
    print("✅ docs/index.html")


HEAD = r"""<title>ViCare-Safe</title>
<meta name="claude:viewport-fit-sides" content="cover">
<meta name="description" content="Tra cứu sức khỏe tiếng Việt: hiểu câu không dấu và teencode, có độ tin cậy, bằng chứng và nguồn.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&family=Lexend:wght@500;600;700&display=swap" rel="stylesheet">
<style>
/* Layout: phiếu kết quả xét nghiệm – hero trên nền giấy kẻ ô, bàn "thử như giám khảo" hai cột, rồi các bảng số liệu */
:root{
  --paper:#f3f6f7; --sheet:#ffffff; --ink:#10242e; --ink-2:#4a5f69; --rule:#d3dee3;
  --accent:#0d6b85; --accent-ink:#ffffff; --accent-wash:#e1eff3;
  --ok:#1b7a4b; --ok-wash:#dcefe4; --warn:#8f5c00; --warn-wash:#f6ead2; --stop:#b3261e; --stop-wash:#f8dfdc;
  --grid:rgba(13,107,133,.08);
  --f-display:"Lexend","Be Vietnam Pro",system-ui,sans-serif;
  --f-text:"Be Vietnam Pro",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#0c171c; --sheet:#12222a; --ink:#e4eef1; --ink-2:#9bb1ba; --rule:#24404b;
    --accent:#5cc2db; --accent-ink:#06222b; --accent-wash:#13343e;
    --ok:#63d39a; --ok-wash:#123425; --warn:#f0c162; --warn-wash:#3a2c0f; --stop:#ff8b80; --stop-wash:#3d1714;
    --grid:rgba(92,194,219,.08); color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --paper:#0c171c; --sheet:#12222a; --ink:#e4eef1; --ink-2:#9bb1ba; --rule:#24404b;
  --accent:#5cc2db; --accent-ink:#06222b; --accent-wash:#13343e;
  --ok:#63d39a; --ok-wash:#123425; --warn:#f0c162; --warn-wash:#3a2c0f; --stop:#ff8b80; --stop-wash:#3d1714;
  --grid:rgba(92,194,219,.08); color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 var(--f-text)}
a{color:var(--accent);text-underline-offset:3px}
a:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.wrap{max-width:1100px;margin:0 auto;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
h1,h2,h3{font-family:var(--f-display);text-wrap:balance;letter-spacing:-.01em;margin:0}
.top{border-bottom:1px solid var(--rule);background:var(--sheet)}
.top .wrap{display:flex;flex-wrap:wrap;align-items:center;gap:8px 20px;padding-block:12px}
.mark{font-family:var(--f-display);font-weight:700;font-size:18px;color:var(--ink);text-decoration:none;display:inline-flex;align-items:center;gap:10px;margin-right:auto}
.mark i{width:18px;height:18px;display:inline-block;position:relative}
.mark i::before,.mark i::after{content:"";position:absolute;background:var(--accent);border-radius:2px}
.mark i::before{left:6px;top:0;width:6px;height:18px}
.mark i::after{left:0;top:6px;width:18px;height:6px}
.top a.l{color:var(--ink-2);text-decoration:none;font-size:15px}
.top a.l:hover{color:var(--ink)}
.hero{background-color:var(--paper);background-image:linear-gradient(var(--grid) 1px,transparent 1px),linear-gradient(90deg,var(--grid) 1px,transparent 1px);background-size:24px 24px;border-bottom:1px solid var(--rule)}
.hero .wrap{padding-block:48px 36px}
.kicker{font-size:13px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--accent)}
.hero h1{font-size:clamp(30px,4.6vw,50px);line-height:1.12;margin:12px 0 14px;max-width:860px}
.lead{font-size:clamp(17px,2vw,19px);color:var(--ink-2);max-width:68ch;margin:0 0 22px}
.btns{display:flex;flex-wrap:wrap;gap:10px}
.btn{display:inline-flex;align-items:center;padding:10px 16px;border-radius:8px;font-weight:600;font-size:15px;text-decoration:none;border:1px solid var(--rule);color:var(--ink);background:var(--sheet)}
.btn.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.warn-line{margin:18px 0 0;font-size:14px;color:var(--ink-2)}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:30px}
.stat{background:var(--sheet);border:1px solid var(--rule);border-radius:10px;padding:14px 16px}
.stat .num{font-family:var(--f-display);font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}
.stat .lbl{color:var(--ink-2);font-size:14px;margin-top:2px}
section .wrap{padding-block:44px}
h2{font-size:clamp(23px,3vw,29px);margin-bottom:8px}
.sub{color:var(--ink-2);margin:0 0 20px;max-width:72ch}
.console{display:grid;grid-template-columns:250px minmax(0,1fr);background:var(--sheet);border:1px solid var(--rule);border-radius:12px;overflow:hidden}
.topics{border-right:1px solid var(--rule);padding:10px;display:flex;flex-direction:column;gap:2px}
.topics .sep{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-2);padding:12px 10px 4px}
.topics button{font:inherit;font-size:15px;text-align:left;cursor:pointer;border:0;background:none;color:var(--ink-2);padding:9px 10px;border-radius:8px}
.topics button:hover{color:var(--ink);background:var(--accent-wash)}
.topics button[aria-selected="true"]{color:var(--ink);background:var(--accent-wash);font-weight:600}
.pane{padding:20px 22px;min-width:0}
.variants{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px}
.variants button{font:inherit;font-size:14px;cursor:pointer;padding:5px 12px;border:1px solid var(--rule);border-radius:999px;background:var(--sheet);color:var(--ink-2)}
.variants button[aria-pressed="true"]{background:var(--ink);border-color:var(--ink);color:var(--paper)}
.label{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-2)}
.q{font-family:var(--f-display);font-size:19px;font-weight:600;margin:2px 0 6px;overflow-wrap:anywhere}
.hist{font-size:14px;color:var(--ink-2);margin:0 0 8px}
.meta{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0 16px}
.chip{font-size:13px;font-weight:600;padding:3px 10px;border-radius:999px;background:var(--accent-wash);color:var(--accent);font-variant-numeric:tabular-nums}
.chip.plain{background:transparent;border:1px solid var(--rule);color:var(--ink-2)}
.t0{background:var(--ok-wash);color:var(--ok)} .t1{background:var(--warn-wash);color:var(--warn)} .t2{background:var(--stop-wash);color:var(--stop)}
.out{border-top:1px dashed var(--rule);padding-top:16px;overflow-wrap:anywhere}
.out p{margin:0 0 12px} .out ul{margin:0 0 12px;padding-left:20px} .out li{margin:5px 0}
.out em{color:var(--ink-2)}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px}
.panel{background:var(--sheet);border:1px solid var(--rule);border-radius:12px;padding:16px 4px 6px;min-width:0}
.panel.wide{grid-column:1/-1}
.panel h3{font-size:17px;padding:0 14px}
.panel p{font-size:14px;color:var(--ink-2);margin:4px 14px 10px}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:15px;font-variant-numeric:tabular-nums}
th,td{padding:9px 14px;border-top:1px solid var(--rule);text-align:left;vertical-align:top;white-space:nowrap}
thead th{font-size:12px;color:var(--ink-2);font-weight:600;text-transform:uppercase;letter-spacing:.05em;border-top:0}
tbody th{font-weight:500;white-space:normal}
td .old{color:var(--ink-2)}
.steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;counter-reset:s;list-style:none;padding:0;margin:0}
.steps li{background:var(--sheet);border:1px solid var(--rule);border-radius:10px;padding:14px 16px}
.steps b{display:block;font-family:var(--f-display);margin-bottom:4px}
.steps b::before{counter-increment:s;content:counter(s) " · ";color:var(--accent);font-variant-numeric:tabular-nums}
.steps span{color:var(--ink-2);font-size:15px}
.limits{margin:0;padding-left:20px;max-width:80ch} .limits li{margin:6px 0}
footer{border-top:1px solid var(--rule);color:var(--ink-2);font-size:14px}
footer .wrap{padding-block:28px 40px}
@media (max-width:880px){
  .stats{grid-template-columns:repeat(2,minmax(0,1fr))}
  .console{grid-template-columns:minmax(0,1fr)}
  .topics{flex-direction:row;overflow-x:auto;border-right:0;border-bottom:1px solid var(--rule)}
  .topics .sep{display:none}
  .topics button{white-space:nowrap}
  .grid2,.steps{grid-template-columns:minmax(0,1fr)}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important;scroll-behavior:auto!important}}
</style>"""

BODY = r"""<header class="top"><div class="wrap">
  <a class="mark" href="#"><i aria-hidden="true"></i>ViCare-Safe</a>
  <a class="l" href="#thu">Thử demo</a>
  <a class="l" href="#ketqua">Kết quả</a>
  <a class="l" href="#cachlam">Cách hoạt động</a>
  <a class="l" href="__GITHUB__">GitHub</a>
</div></header>

<div class="hero"><div class="wrap">
  <div class="kicker">AI Challenge 24h · AI4LIFE – ĐH Bách Khoa Hà Nội &amp; Rikkei</div>
  <h1>Tra cứu sức khỏe tiếng Việt, hiểu cả khi bạn gõ không dấu.</h1>
  <p class="lead">ViCare-Safe là trợ lý <strong>hỗ trợ quyết định</strong>: trích nguyên văn lời bác sĩ trong kho tài liệu, kèm
  <strong>độ tin cậy</strong>, <strong>bằng chứng</strong> và <strong>nguồn</strong>. Không chẩn đoán, không kê đơn, và nói rõ khi chưa đủ căn cứ.</p>
  <div class="btns">
    <a class="btn primary" id="live" href="#thu">Thử demo bên dưới</a>
    <a class="btn" href="__COLAB__">Mở notebook trên Colab</a>
    <a class="btn" href="report.html">Báo cáo đánh giá</a>
    <a class="btn" href="__GITHUB__">Mã nguồn</a>
  </div>
  <p class="warn-line">Demo học thuật, không thay thế tư vấn của nhân viên y tế. Cấp cứu: gọi <strong>115</strong>.</p>
  <div class="stats">__STATS__</div>
</div></div>

<section id="thu"><div class="wrap">
  <h2>Thử như giám khảo</h2>
  <p class="sub">Chọn một câu hỏi rồi đổi cách gõ: có dấu, không dấu, teencode. Đây là đầu ra thật của hệ thống (chế độ trích xuất,
  không LLM), được tính sẵn để trang xem được ở mọi nơi.</p>
  <div class="console">
    <div class="topics" role="tablist" aria-label="Câu hỏi mẫu" id="topics"></div>
    <div class="pane">
      <div class="variants" id="variants" aria-label="Cách gõ"></div>
      <div class="label">Người dùng hỏi</div>
      <p class="hist" id="hist" hidden></p>
      <p class="q" id="q"></p>
      <div class="meta" id="meta"></div>
      <div class="out" id="out" aria-live="polite"></div>
    </div>
  </div>
</div></section>

<section id="ketqua"><div class="wrap">
  <h2>Kết quả đo được</h2>
  <p class="sub">Cùng kho tài liệu và cùng 100 câu dev của notebook Ban tổ chức. Baseline là notebook gốc (tìm kiếm vector e5).
  Cấu hình giống demo trên CPU: reranker chấm __NCAND__ ứng viên. Khoảng tin cậy 95% và kiểm định McNemar có trong
  <a href="report.html">báo cáo</a>.</p>
  <div class="grid2">
    <div class="panel wide">
      <h3>Tìm đúng tài liệu: baseline → ViCare-Safe</h3>
      <div class="scroll"><table><thead><tr><th>Cách gõ</th><th>Recall@1</th><th>Recall@5</th><th>Recall@10</th><th>MRR@10</th></tr></thead>
      <tbody>__RET__</tbody></table></div>
    </div>
    <div class="panel">
      <h3>Độ tin cậy có nói thật không?</h3>
      <p>Tỷ lệ tài liệu gốc nằm trong 3 nguồn hiển thị, theo mức ViCare tự báo. ECE = __ECE__ · AUROC = __AUROC__.</p>
      <div class="scroll"><table><thead><tr><th>Mức</th><th>Số câu</th><th>Dự đoán</th><th>Thực tế</th></tr></thead><tbody>__REL__</tbody></table></div>
    </div>
    <div class="panel">
      <h3>An toàn</h3>
      <p>Bộ câu tự viết (20 ca cấp cứu, 15 câu thường, 10 câu hỏi liều), nên kết quả có thể lạc quan.</p>
      <div class="scroll"><table><thead><tr><th>Kiểm tra</th><th>Baseline</th><th>ViCare-Safe</th></tr></thead><tbody>__SAF__</tbody></table></div>
    </div>
    <div class="panel wide">
      <h3>ViCare quyết định thế nào?</h3>
      <p>Câu ngoài phạm vi bị từ chối khi điểm reranker dưới τ = __TAU__. Còn lại chia mức Cao, Trung bình, Thấp; mức Thấp không trả lời, chỉ hiện nguồn.</p>
      <div class="scroll"><table><thead><tr><th>Bộ câu hỏi</th><th>Cao</th><th>Trung bình</th><th>Thấp</th><th>Từ chối</th></tr></thead><tbody>__DEC__</tbody></table></div>
    </div>
  </div>
</div></section>

<section id="cachlam"><div class="wrap">
  <h2>Cách hoạt động</h2>
  <p class="sub">Sáu bước chạy theo thứ tự cho mỗi câu hỏi. Không cần GPU hay LLM, chạy được trên CPU miễn phí.</p>
  <ol class="steps">
    <li><b>Lớp an toàn</b><span>Nhận ra tình huống cấp cứu (có dấu lẫn không dấu) để nhắc gọi 115; chặn câu hỏi xin liều thuốc, kê đơn.</span></li>
    <li><b>Chuẩn hóa teencode</b><span>“ko, j, dc, bs, bv” thành dạng chuẩn, có luật tránh nhầm như vitamin K, viêm gan C, ng/mL.</span></li>
    <li><b>Khôi phục dấu</b><span>Mô hình trigram học từ chính kho tài liệu và giải mã Viterbi: “con toi bi sot” thành “con tôi bị sốt”, khoảng 2 ms.</span></li>
    <li><b>Truy xuất lai</b><span>BM25 trên chữ bỏ dấu cộng vector e5, gộp bằng RRF, rồi cross-encoder bge-reranker-v2-m3 xếp hạng lại.</span></li>
    <li><b>Độ tin cậy</b><span>Hồi quy logistic trên 7 tín hiệu truy xuất, ước lượng khả năng tài liệu đúng nằm trong 3 nguồn hiển thị.</span></li>
    <li><b>Bằng chứng và nguồn</b><span>Chấm từng câu trong tài liệu, trích nguyên văn câu liên quan nhất kèm số nguồn [n].</span></li>
  </ol>
</div></section>

<section><div class="wrap">
  <h2>Giới hạn cần biết</h2>
  <ul class="limits">
    <li><strong>Độ tin cậy</strong> là độ tin vào việc tìm đúng tài liệu, không phải độ đúng y khoa.</li>
    <li>Dữ liệu ViHealthQA (Vinmec, VnExpress) thu thập khoảng 2019–2022, có thể đã cũ và chỉ dùng cho nghiên cứu.</li>
    <li>Câu không dấu và teencode trong đánh giá được tạo tự động từ câu gốc; bộ câu cấp cứu do nhóm tự viết.</li>
    <li>Phát hiện cấp cứu dựa trên từ khóa. Luôn gọi 115 khi nghi ngờ cấp cứu.</li>
  </ul>
</div></section>

<footer><div class="wrap">
  Dữ liệu: Nguyen et al., <em>SPBERTQA: A Two-Stage Question Answering System Based on Sentence Transformers for Medical Texts</em>,
  KSEM 2022 · <a href="https://huggingface.co/datasets/tarudesu/ViHealthQA">tarudesu/ViHealthQA</a> ·
  Mã nguồn MIT: <a href="__GITHUB__">github.com/chinguyenkhac/ViCare</a>
</div></footer>

<script>
const DATA = __DATA__;
const SPECIAL = new Set(["oos","inject","dose","falseref","emerg","followup"]);
const TIER = {"Cao":0,"Trung bình":1,"Thấp":2};
const STATUS = {answered:"Đã trả lời",low_confidence:"Độ tin cậy thấp: chỉ hiện nguồn",refused_oos:"Từ chối: ngoài phạm vi",dose_blocked:"Chặn hỏi liều"};
let cur = {g: DATA.order[0], v: 0};
const $ = id => document.getElementById(id);
function renderTopics(){
  const t = $("topics"); t.textContent = "";
  let sep = false;
  DATA.order.forEach(g => {
    if (SPECIAL.has(g) && !sep){ const s = document.createElement("div"); s.className = "sep"; s.textContent = "Tình huống đặc biệt"; t.appendChild(s); sep = true; }
    const b = document.createElement("button");
    b.type = "button"; b.setAttribute("role","tab"); b.textContent = DATA.groups[g].title;
    b.setAttribute("aria-selected", String(g === cur.g));
    b.addEventListener("click", () => { cur = {g, v: 0}; render(); });
    t.appendChild(b);
  });
}
function chip(text, cls){ const s = document.createElement("span"); s.className = "chip " + (cls || ""); s.textContent = text; return s; }
function render(){
  renderTopics();
  const grp = DATA.groups[cur.g], it = grp.items[cur.v];
  const vs = $("variants"); vs.textContent = "";
  if (grp.items.length > 1) grp.items.forEach((x, i) => {
    const b = document.createElement("button"); b.type = "button"; b.textContent = x.variant;
    b.setAttribute("aria-pressed", String(i === cur.v));
    b.addEventListener("click", () => { cur.v = i; render(); }); vs.appendChild(b);
  });
  vs.hidden = grp.items.length <= 1;
  $("q").textContent = it.question;
  const h = $("hist");
  if (it.history){ h.hidden = false; h.textContent = "Câu trước: " + it.history[it.history.length - 1]; } else { h.hidden = true; }
  const meta = $("meta"); meta.textContent = "";
  meta.appendChild(chip(STATUS[it.status] || it.status, "plain"));
  if (it.level) meta.appendChild(chip("Độ tin cậy " + it.level + " · " + Math.round(it.prob * 100) + "%", "t" + TIER[it.level]));
  if (it.emergency) meta.appendChild(chip("Cảnh báo cấp cứu", "t2"));
  $("out").innerHTML = it.html;
}
render();
</script>"""

PAGES_EXTRA = r"""<script>
fetch("space.json").then(r => r.ok ? r.json() : null).then(j => {
  if (j && j.url){ const a = document.getElementById("live"); a.href = j.url; a.textContent = "Mở demo chat trực tiếp"; }
}).catch(() => {});
</script>"""


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else None)
