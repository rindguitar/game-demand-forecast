"""
フィルタリング機能のテストスクリプト

フィルタリング前後を比較して、実際に非英語レビューが除外されているか確認する
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import requests
import time
from src.data.steam_collector import get_steam_reviews, is_valid_english_review


def get_raw_reviews_no_filter(app_id: int, num: int = 100):
    """フィルタリングなしでSteam APIから直接レビューを取得"""
    base_url = "https://store.steampowered.com/appreviews/"
    params = {
        'json': 1,
        'language': 'all',  # 全言語取得
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


def main():
    """フィルタリング機能のテスト"""

    print("=" * 80)
    print("フィルタリング機能テスト（200件収集）")
    print("=" * 80)

    # Baldur's Gate 3 から200件収集（フィルタリング前後を比較）
    app_id = 1086940
    num = 200

    print(f"\nゲーム: Baldur's Gate 3 (app_id={app_id})")
    print(f"目標件数: {num}件")

    # 1. フィルタリングなしで取得
    print("\n[1/2] フィルタリングなしで取得中...")
    raw_reviews = get_raw_reviews_no_filter(app_id, num)
    print(f"✅ 取得完了: {len(raw_reviews)}件（全言語）")

    # フィルタリング前の統計
    print("\n【フィルタリング前の統計】")
    filtered_out = []
    valid_count = 0

    for review in raw_reviews:
        text = review['review_text']
        if is_valid_english_review(text):
            valid_count += 1
        else:
            filtered_out.append(review)

    print(f"有効な英語レビュー: {valid_count}件 ({valid_count/len(raw_reviews)*100:.1f}%)")
    print(f"除外される: {len(filtered_out)}件 ({len(filtered_out)/len(raw_reviews)*100:.1f}%)")

    # 除外されるレビューのサンプル表示
    print("\n【除外されるレビューのサンプル（先頭10件）】")
    for i, review in enumerate(filtered_out[:10], 1):
        text = review['review_text']
        lang = review['language']
        reason = []

        # ASCII判定
        if not all(ord(c) < 128 for c in text):
            reason.append("非ASCII")

        # 長さ判定
        if len(text) < 20:
            reason.append(f"短文({len(text)}文字)")

        # アルファベット割合判定
        alpha_chars = sum(1 for c in text if c.isalpha())
        if len(text) > 0 and alpha_chars / len(text) < 0.5:
            reason.append(f"アルファベット{alpha_chars/len(text)*100:.0f}%")

        reason_str = ", ".join(reason)
        preview = text[:80] if len(text) > 80 else text
        print(f"{i}. [{lang}] [{reason_str}] {preview}")

    # 2. フィルタリングありで取得
    print(f"\n[2/2] フィルタリングありで{num}件収集中...")
    filtered_reviews = get_steam_reviews(
        app_id=app_id,
        language='english',
        review_type='all',
        num=num
    )

    print(f"\n✅ 収集完了: {len(filtered_reviews)}件")

    # 検証: 全レビューがフィルタリング条件を満たしているか
    print("\n【フィルタリング後の検証】")

    non_ascii_count = 0
    short_count = 0
    low_alpha_count = 0

    for review in filtered_reviews:
        text = review['review_text']

        # ASCII判定
        if not all(ord(c) < 128 for c in text):
            non_ascii_count += 1

        # 長さ判定
        if len(text) < 20:
            short_count += 1

        # アルファベット割合判定
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars / len(text) < 0.5:
            low_alpha_count += 1

    print(f"非ASCII文字を含む: {non_ascii_count}件 (期待値: 0)")
    print(f"20文字未満: {short_count}件 (期待値: 0)")
    print(f"アルファベット50%未満: {low_alpha_count}件 (期待値: 0)")

    if non_ascii_count == 0 and short_count == 0 and low_alpha_count == 0:
        print("\n✅ フィルタリング成功: 全レビューが条件を満たしています")
    else:
        print("\n⚠️ フィルタリング失敗: 一部レビューが条件を満たしていません")

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
