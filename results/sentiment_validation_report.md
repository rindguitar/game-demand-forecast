# 感情分析精度検証レポート（Issue #5）

**日付**: 2026-04-08
**対象**: 英語レビュー100件（Positive 50件 + Negative 50件）
**モデル**: DistilBERT事前学習済みモデル（`distilbert-base-uncased-finetuned-sst-2-english`）

---

## 📊 検証結果サマリー

| 指標 | 結果 | 目標 | 判定 |
|------|------|------|------|
| **Accuracy** | **79.00%** | 90%以上 | ❌ 未達 |
| Precision | 91.43% | - | ✅ 良好 |
| Recall | 64.00% | - | ⚠️ 低い |
| F1 Score | 75.29% | - | ⚠️ 低い |

### クラス別性能

| クラス | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| **Negative (0)** | 0.72 | **0.94** | 0.82 | 50 |
| **Positive (1)** | 0.91 | **0.64** | 0.75 | 50 |

**混同行列:**
```
                 Predicted
                 Neg    Pos
Actual  Neg    [  47] [   3]
        Pos    [  18] [  32]
```

**問題点:**
- Negative検出: 94%（良好）
- **Positive検出: 64%（低い）** ← 主な問題

---

## 🔍 原因分析

### Steamレビューの文化的特性

Steamレビューには独特の文化があり、「おすすめ（Recommended）」でも**皮肉や毒舌を含む表現**が多用されます。

**具体例（収集したデータより）:**

#### ケース1: 皮肉を含む「おすすめ」
```
レビュー: "Installed this game to have fun. Now I study human psychology,
           anger management, and advanced muting techniques.
           10/10 free therapy (you pay with your sanity)."
Steamラベル: おすすめ (game_rating=1)
DistilBERT予測: NEGATIVE（ネガティブな表現を検出）
```

#### ケース2: 毒舌を含む「おすすめ」
```
レビュー: "The game you love to hate, hate to love.
           Playing this game will bring out the worst in you..."
Steamラベル: おすすめ (game_rating=1)
DistilBERT予測: NEGATIVE（"worst" などのネガティブワードを検出）
```

#### ケース3: 短いネガティブワードで「おすすめ」
```
レビュー: "toxic‍"
Steamラベル: おすすめ (game_rating=1)
DistilBERT予測: NEGATIVE（明らかにネガティブ）
```

### なぜこのようなレビューが「おすすめ」なのか？

1. **愛憎表現**: プレイヤーはゲームを愛しているが、中毒性やストレスを皮肉で表現
2. **コミュニティ文化**: Steamでは毒舌レビューが一種のユーモアとして受け入れられている
3. **複雑な感情**: 「ゲームは面白いがストレスフル」という複雑な感情を表現

### 事前学習済みモデルの限界

- **一般的な感情分析**: 映画レビュー、商品レビューなどで学習
- **Steam特有の表現**: 学習していない
- **文脈理解の不足**: 皮肉や複雑な感情を理解できない

---

## ✅ 達成できたこと

### 1. 実装の正しさを確認
- データ収集機能: ✅ 正常動作
- 感情分析機能: ✅ 正常動作
- 評価機能: ✅ 正常動作
- **実装にバグなし**

### 2. 課題の明確化
- 事前学習済みモデルはSteamレビューに不十分（79%）
- **ファインチューニングが必要**

### 3. 次のステップの方向性確立
- Issue #6でSteam専用にDistilBERTをファインチューニング
- 目標精度: **85-90%**

---

## 🎯 結論と推奨事項

### Issue #5の結論
- ✅ **実装確認完了**: コードは正常に動作
- ⚠️ **精度未達**: 事前学習済みモデルでは79%
- ✅ **課題明確化**: Steam特有の表現に対応する必要性を確認

### 次のステップ（Issue #6）
1. **データセット拡大**: 100件 → **1000件**（Positive 500件 + Negative 500件）
2. **ファインチューニング**: PyTorchでDistilBERTをSteamレビューで再学習
3. **目標精度**: **85-90%**（Steam特有の表現を学習）

### 言語選択
- **英語で確定**
- 理由:
  - データセット規模: 英語3.1M+ vs 日本語少量
  - モデル品質: 英語モデルの方が成熟
  - 一般化可能性: より広範な市場分析が可能

---

## 📁 成果物

1. **実装コード**:
   - `src/data/steam_collector.py`: Steam APIデータ収集
   - `src/data/preprocessing.py`: データ前処理
   - `src/nlp/sentiment.py`: 感情分析機能
   - `src/nlp/evaluation.py`: 精度評価機能

2. **テストスクリプト**:
   - `scripts/test_sentiment.py`: 感情分析機能テスト（6/6成功）
   - `scripts/validate_sentiment_english.py`: 100件精度検証

3. **データ**:
   - `data/validation/english_reviews_100.csv`: 検証用データ100件

4. **レポート**:
   - `results/sentiment_validation_report.md`: 本レポート

---

## 🔗 参考資料

- [Wiki: Language Selection Analysis](https://github.com/rindguitar/game-demand-forecast/wiki/Language-Selection-Analysis)
- [Wiki: Game Review Datasets](https://github.com/rindguitar/game-demand-forecast/wiki/Game-Review-Datasets)
- [Issue #5: 感情分析の精度検証](https://github.com/rindguitar/game-demand-forecast/issues/5)
- [Issue #6: PyTorchで感情分析モデルをゼロから学習](https://github.com/rindguitar/game-demand-forecast/issues/6)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
