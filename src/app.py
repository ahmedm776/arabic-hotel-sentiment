"""FastAPI app serving the saved HARD sentiment pipeline. Run: uvicorn src.app:app --port 8000"""
import os, joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.prepare import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = joblib.load(os.path.join(HERE, "..", "data", "model.joblib"))


def _featurize(texts):
    # word+char union matching src/train.py (selected by hotel-grouped train CV)
    from scipy.sparse import hstack
    return hstack([BUNDLE["vectorizer"].transform(texts),
                   BUNDLE["vectorizer_char"].transform(texts)]).tocsr()

app = FastAPI(title="Arabic Hotel Review Sentiment (HARD)")

class Review(BaseModel):
    text: str

PAGE = """<!doctype html><html dir="rtl" lang="ar"><head><meta charset="utf-8">
<title>تحليل مشاعر تقييمات الفنادق</title>
<style>body{font-family:sans-serif;max-width:640px;margin:40px auto;padding:0 16px}
textarea{width:100%;height:120px;font-size:16px}button{font-size:16px;padding:8px 20px;margin-top:8px}
#out{margin-top:16px;font-size:18px;padding:12px;border-radius:8px;background:#f2f2f2}</style></head>
<body><h2>تحليل مشاعر تقييمات الفنادق (HARD)</h2>
<p>نموذج لوجستي + TF-IDF (كلمات وحروف) مدرّب على ٨٢ ألف تقييم من فنادق، ومقيّم على فنادق أخرى لم تدخل في التدريب (مجموعة الاختبار نفسها أُعيد استخدامها في تقييم ثانٍ بعد تحسين النموذج).</p>
<textarea id="t" placeholder="اكتب تقييم فندق بالعربية..."></textarea><br>
<button onclick="go()">تحليل</button><div id="out"></div>
<script>
async function go(){const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({text:document.getElementById('t').value})});const j=await r.json();
document.getElementById('out').textContent=r.ok?(j.label+' - درجة النموذج للإيجابية: '+(j.probability*100).toFixed(1)+'% (درجة غير معايرة، ليست ضمانًا لصحة التصنيف)'):('خطأ: '+j.detail);}
</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE

@app.get("/demo", response_class=HTMLResponse)
def demo(text: str = ""):
    # Renders the UI with a submitted review and its result - also used for visual checks.
    if not text.strip():
        result = '<div id="out">خطأ: النص فارغ بعد التنظيف</div>'
        filled = ""
    else:
        p = float(BUNDLE["model"].predict_proba(_featurize([normalize(text)]))[0, 1])
        label = "إيجابي" if p >= 0.5 else "سلبي"
        result = f'<div id="out">{label} - درجة النموذج للإيجابية: {p*100:.1f}% (درجة غير معايرة، ليست ضمانًا لصحة التصنيف)</div>'
        filled = text.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
    return PAGE.replace('<textarea id="t" placeholder=', f'<textarea id="t" placeholder=', 1).replace(
        '></textarea><br>', f'>{filled}</textarea><br>').replace('<div id="out"></div>', result)

@app.post("/predict")
def predict(r: Review):
    text = normalize(r.text)
    if not text:
        return {"detail": "empty review after normalization"}
    p = float(BUNDLE["model"].predict_proba(_featurize([text]))[0, 1])
    return {"label": "إيجابي" if p >= 0.5 else "سلبي", "probability": round(p, 4),
            "note": "model trained on 2016 Booking.com reviews; probability is the uncalibrated positive-class score from logistic regression, not a guarantee of correctness - see metrics.json"}
