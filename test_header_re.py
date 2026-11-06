"""Plain-assert tests for header_re. Run: python3 test_header_re.py  (exit 0 = pass)."""
from header_re import (RAW_HEADER_RE, NORM_HEADER_RE, RAW_DESCRIPTOR_FORMS,
                       NORM_DESCRIPTOR_FORMS, normalize, strip_header_raw, preprocess)

def t(name, cond):
    assert cond, name
    print("pass:", name)

# 1. every raw form + period is stripped from raw text
for f in RAW_DESCRIPTOR_FORMS:
    s, hit = strip_header_raw(f + ". الفندق كان نظيف")
    t(f"raw strip {f}", hit and s == "الفندق كان نظيف")
# 2. after strip+normalize NO normalized-form header remains (all derived forms)
for f in RAW_DESCRIPTOR_FORMS:
    out = preprocess(f + ". الفندق كان نظيف")
    t(f"no header after preprocess {f}", not NORM_HEADER_RE.match(out))
    t(f"content kept {f}", out.startswith(normalize("الفندق")))
# 3. longest match: 'جيد جداً.' must strip the whole form, not leave 'جداً.'
out = preprocess("جيد جداً. موقع ممتاز")
t("longest match جيد جداً", not out.startswith("جدا") and out == normalize("موقع ممتاز"))
out = preprocess("ضعيف جداً. خدمة سيئة")
t("longest match ضعيف جداً", out == normalize("خدمة سيئة"))
# 4. period boundary: descriptor without a period is NOT stripped
s, hit = strip_header_raw("جيد الفندق بصراحة")
t("no period no strip", not hit and s.startswith("جيد الفندق"))
s, hit = strip_header_raw("ممتاز")
t("bare descriptor no period no strip", not hit)
# 5. position boundary: descriptor not at position 0 is NOT stripped
s, hit = strip_header_raw("الفندق جيد. لكن السعر عالي")
t("mid-text descriptor untouched", not hit)
# 6. header-only review becomes empty after strip (dropped downstream)
s, hit = strip_header_raw("استثنائي.")
t("header-only -> empty", hit and s.strip() == "")
# 7. hamza/diacritic raw variants strip and normalize identically
t("مخيب للأمل variant", preprocess("مخيب للأمل. تجربة سيئة") == normalize("تجربة سيئة"))
t("normalized-form derivation covers hamza-yaa", "استثنايي" in NORM_DESCRIPTOR_FORMS and "رايع" in NORM_DESCRIPTOR_FORMS)
# 8. pattern parity with audit r2 (byte-identical raw pattern)
t("raw pattern identical to audit r2",
  RAW_HEADER_RE.pattern == r"^(استثنائي|رائع|جيد جداً|جيد جدا|جيد|مقبول|ضعيف جداً|ضعيف جدا|ضعيف|مخيب للأمل|ممتاز)\.\s*")
print("ALL TESTS PASSED")

# 9. double headers strip fully
from header_re import strip_header_raw_all
s, n = strip_header_raw_all("ممتاز. ممتاز. لا شي")
assert n == 2 and s == "لا شي", (n, s)
s, n = strip_header_raw_all("جيد جداً. جيد. موقع لطيف")
assert n == 2 and not NORM_HEADER_RE.match(normalize(s)), (n, s)
s, n = strip_header_raw_all("ممتاز. لا شي")
assert n == 1 and s == "لا شي", (n, s)
print("pass: double-header stripping")

# 10. mixed orthography double header strips fully, header-only becomes empty
from header_re import preprocess as pp
assert pp("ممتاز. رايع.") == "", repr(pp("ممتاز. رايع."))
assert pp("ممتاز. رايع. لا شي") == normalize("لا شي")
assert pp("رايع. فندق هادئ") == normalize("فندق هادئ")
print("pass: mixed-orthography layer stripping")
