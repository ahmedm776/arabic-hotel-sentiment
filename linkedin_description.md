# LinkedIn description (draft)

**Arabic Hotel Review Sentiment Analysis** - Python, scikit-learn, FastAPI

Built an end-to-end Arabic NLP classifier on the HARD dataset (105k hotel reviews
from Booking.com, Elnagar et al. 2018, cited; no explicit license file - used with
attribution for educational purposes).

- Honest evaluation design: group split by hotel, so the test set contains only
  hotels the model never saw during training; TF-IDF fit on training data only;
  protocol frozen before fitting; single evaluation, all metrics reported as-is.
- Logistic regression over word 1-2-gram TF-IDF: 93.9% accuracy, 0.979 AUC on the
  held-out hotels, vs 48.1% majority baseline and 91.1% for MultinomialNB.
- Light Arabic normalization (diacritics, tatweel, alef/yaa variants) without
  external lexicons or stemming.
- FastAPI app with an Arabic (RTL) interface; pytest suite covering split
  integrity, label mapping, normalization, and prediction sanity.
- Limitations documented: 2016 data, star-rating-derived labels, single frozen split.
