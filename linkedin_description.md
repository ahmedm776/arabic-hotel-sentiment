# Arabic Hotel Review Sentiment

Arabic NLP with word/character TF-IDF and logistic regression. Corrected preprocessing
removes label-leaking rating headers before normalization. Hotel-disjoint split with
zero normalized-text overlap after cleaning: 80,565 train / 22,488 test.
Accuracy 88.06%, F1 0.8843, AUC 0.9458 on the reused development benchmark,
not independent confirmation. Earlier 94.62% accuracy is withdrawn due to leakage.
FastAPI RTL demo; synthetic preprocessing regression assertions. Raw reviews not included.
