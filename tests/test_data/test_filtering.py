"""
フィルタリング機能のテストスクリプト

フィルタリング前後を比較して、実際に非英語レビューが除外されているか確認する。
langdetectによるフィリピン語・スペイン語等のラテン文字言語の除外も検証する。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import requests
import time
from langdetect import detect_langs, LangDetectException, DetectorFactory
DetectorFactory.seed = 0  # 再現性のために固定
from src.data.steam_collector import get_steam_reviews


def get_raw_reviews_no_filter(app_id: int, num: int = 100):
    """フィルタリングなしでSteam APIから直接レビューを取得"""
    base_url = "https://store.steampowered.com/appreviews/"
    params = {
        'json': 1,
        'language': 'all',
        'filter': 'recent',
        'review_type': 'all',
        'purchase_type': 'all',
        'num_per_page': min(100, num),
    }

    reviews = []
    cursor = '*'

    while len(reviews) < num:
        params['cursor'] = cursor
        response = requests.get(f"{base_url}{app_id}", params=params, timeout=10)
        data = response.json()

        if data.get('success') != 1:
            break

        api_reviews = data.get('reviews', [])
        if not api_reviews:
            break

        for review in api_reviews:
            if len(reviews) >= num:
                break
            reviews.append({
                'review_text': review.get('review', ''),
                'language': review.get('language', ''),
            })

        cursor = data.get('cursor')
        if not cursor:
            break
        time.sleep(0.5)

    return reviews


def get_rejection_reason(text: str, min_length: int = 20, lang_confidence: float = 0.8) -> list:
    """除外理由を返す"""
    reasons = []

    if not all(ord(c) < 128 for c in text):
        reasons.append("非ASCII")

    if len(text) < min_length:
        reasons.append(f"短文({len(text)}文字)")

    alpha_chars = sum(1 for c in text if c.isalpha())
    if len(text) > 0 and alpha_chars / len(text) < 0.5:
        reasons.append(f"アルファベット{alpha_chars/len(text)*100:.0f}%")

    # ASCIIかつmin_length以上かつアルファベット50%以上のものはlangdetectで判定
    if (all(ord(c) < 128 for c in text) and
            len(text) >= min_length and
            len(text) > 0 and
            sum(1 for c in text if c.isalpha()) / len(text) >= 0.5):
        try:
            langs = detect_langs(text)
            en_score = next((l.prob for l in langs if l.lang == 'en'), 0.0)
            if en_score < lang_confidence:
                top_lang = langs[0].lang if langs else 'unknown'
                reasons.append(f"langdetect:en={en_score:.2f}({top_lang})")
        except LangDetectException:
            reasons.append("langdetect:判定不可")

    return reasons


def main():
    """フィルタリング機能のテスト"""

    print("=" * 80)
    print("フィルタリング機能テスト（200件収集）")
    print("=" * 80)

    app_id = 1086940
    num = 200
    min_length = 20
    lang_confidence = 0.8  # 英語信頼スコア閾値

    print(f"\nゲーム: Baldur's Gate 3 (app_id={app_id})")
    print(f"目標件数: {num}件")
    print(f"最小文字数: {min_length}文字")
    print(f"langdetect信頼スコア閾値: {lang_confidence}")

    # 1. フィルタリングなしで取得
    print("\n[1/2] フィルタリングなしで取得中（全言語）...")
    raw_reviews = get_raw_reviews_no_filter(app_id, num)
    print(f"✅ 取得完了: {len(raw_reviews)}件")

    # 各フィルター段階での除外数を集計
    print("\n【フィルタリング段階別の統計】")

    non_ascii_out = 0
    short_out = 0
    low_alpha_out = 0
    langdetect_out = 0
    valid_count = 0

    langdetect_samples = []

    for review in raw_reviews:
        text = review['review_text']
        lang = review['language']

        # 段階別チェック
        if not all(ord(c) < 128 for c in text):
            non_ascii_out += 1
            continue

        if len(text) < min_length:
            short_out += 1
            continue

        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars / len(text) < 0.5:
            low_alpha_out += 1
            continue

        # langdetectチェック（信頼スコア閾値）
        try:
            langs = detect_langs(text)
            en_score = next((l.prob for l in langs if l.lang == 'en'), 0.0)
            if en_score < lang_confidence:
                langdetect_out += 1
                if len(langdetect_samples) < 5:
                    top_lang = langs[0].lang if langs else 'unknown'
                    langdetect_samples.append({
                        'text': text,
                        'steam_lang': lang,
                        'en_score': en_score,
                        'detected': top_lang
                    })
                continue
        except LangDetectException:
            langdetect_out += 1
            continue

        valid_count += 1

    total = len(raw_reviews)
    print(f"非ASCII（中国語・ロシア語等）: {non_ascii_out}件 ({non_ascii_out/total*100:.1f}%)")
    print(f"短文（{min_length}文字未満）          : {short_out}件 ({short_out/total*100:.1f}%)")
    print(f"アルファベット50%未満       : {low_alpha_out}件 ({low_alpha_out/total*100:.1f}%)")
    print(f"langdetect（非英語と判定）  : {langdetect_out}件 ({langdetect_out/total*100:.1f}%)")
    print(f"有効な英語レビュー          : {valid_count}件 ({valid_count/total*100:.1f}%)")

    # langdetectで除外されたサンプル
    if langdetect_samples:
        print(f"\n【langdetectで除外されたサンプル（信頼スコア{lang_confidence}未満）】")
        for i, sample in enumerate(langdetect_samples, 1):
            preview = sample['text'][:100] if len(sample['text']) > 100 else sample['text']
            print(f"{i}. [Steam:{sample['steam_lang']}] [en={sample['en_score']:.2f}, 検出:{sample['detected']}] {preview}")
    else:
        print(f"\n【langdetectで除外されたサンプル】")
        print("  なし（全てのラテン文字レビューが英語と判定されました）")

    # 2. フィルタリングありで取得
    print(f"\n[2/2] フィルタリングありで{num}件収集中...")
    filtered_reviews = get_steam_reviews(
        app_id=app_id,
        language='english',
        review_type='all',
        num=num
    )

    print(f"\n✅ 収集完了: {len(filtered_reviews)}件")

    # 検証
    print("\n【フィルタリング後の検証】")
    issues = []
    for review in filtered_reviews:
        text = review['review_text']
        reasons = get_rejection_reason(text, min_length, lang_confidence)
        if reasons:
            issues.append((text, reasons))

    if not issues:
        print("✅ フィルタリング成功: 全レビューが条件を満たしています")
    else:
        print(f"⚠️ フィルタリング失敗: {len(issues)}件が条件を満たしていません")
        for text, reasons in issues[:5]:
            print(f"  - [{', '.join(reasons)}] {text[:80]}")

    # サンプル表示
    print("\n【有効な英語レビューのサンプル（先頭5件）】")
    for i, review in enumerate(filtered_reviews[:5], 1):
        text = review['review_text']
        preview = text[:100] if len(text) > 100 else text
        print(f"{i}. [{len(text)}文字] {preview}...")

    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80)


if __name__ == '__main__':
    main()
