"""
DAPT用 未ラベルコーパス収集スクリプト

多様なゲームの生レビューテキストを大量に集める（DAPTのMLM継続学習用）。
MLMはラベル不要・P/N均等不要なので、雑に大量に集めるだけでよい。

設計:
- 母集団: Steam公式検索APIのレビュー数順ゲーム（多様なゲームから幅広い言い回しを収集）
- 除外: OODテストの20ゲーム＋学習データの7ゲーム（両方のCSVからgame_idを動的に読む）
  - OODゲーム: 評価リーケージ防止
  - 学習ゲーム: 10kは train/val/test に分割されtest split(約15%)を含むため、DAPTが
    間接的にそのtest文を"見て"in-domain Test Accを水増しするのを防ぐ
- get_steam_reviews が内部で有効な英語レビューのみ抽出する
- 途中クラッシュに備え、一定件数ごとにCSVを上書き保存（収集のやり直しを防ぐ）

出力: data/dapt/corpus.csv（review_text, game_id）
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd

from src.data.steam_collector import get_popular_games, get_steam_reviews


def load_excluded_game_ids(csv_paths: list) -> set:
    """指定CSV群に登場するゲームIDを集める（リーケージ防止用の除外集合）

    OODテストCSV（評価リーケージ防止）と学習CSV（10k内のtest split混入防止）の
    両方を渡す。各CSVの game_id 列を読んで和集合をとる。
    """
    excluded = set()
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"  ⚠️ CSVが見つからない（{path}）→ スキップ")
            continue
        df = pd.read_csv(path)
        if 'game_id' in df.columns:
            excluded |= set(int(g) for g in df['game_id'].dropna().unique())
    return excluded


def save_corpus(rows: list, output: str):
    """収集済みレビューをCSVに保存（上書き）"""
    os.makedirs(os.path.dirname(output), exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='DAPT用 未ラベルコーパス収集')
    parser.add_argument('--target', type=int, default=100000, help='収集する総レビュー数')
    parser.add_argument('--per-game', type=int, default=200, help='1ゲームあたり収集数')
    parser.add_argument('--n-pages', type=int, default=40, help='母集団取得ページ数（多めに確保）')
    parser.add_argument('--output', type=str, default='data/dapt/corpus.csv', help='出力先CSV')
    parser.add_argument('--ood-csv', type=str, default='data/test/reviews_ood_2000.csv',
                        help='除外するOODゲームを読むCSV（評価リーケージ防止）')
    parser.add_argument('--train-csv', type=str, default='data/train/reviews_10000.csv',
                        help='除外する学習ゲームを読むCSV（10k内のtest split混入防止）')
    parser.add_argument('--save-every', type=int, default=5000,
                        help='この件数ごとに途中保存（クラッシュ対策）')
    args = parser.parse_args()

    print("=" * 60)
    print("DAPT用 未ラベルコーパス収集")
    print("=" * 60)
    print(f"  目標: {args.target:,}件 / 1ゲーム最大 {args.per_game}件")

    # 1. 除外ゲーム（OODテスト＋学習データ）
    excluded = load_excluded_game_ids([args.ood_csv, args.train_csv])
    print(f"  除外ゲーム（OODテスト＋学習）: {len(excluded)}本")

    # 2. 母集団取得
    print(f"\n[1/2] 母集団取得（{args.n_pages}ページ）...")
    games = get_popular_games(n_pages=args.n_pages)
    candidates = [(aid, name) for aid, name in games if aid not in excluded]
    print(f"  取得 {len(games)}本 → 除外後 {len(candidates)}本")

    # 3. 各ゲームから生レビューを収集（ラベル不要）
    print(f"\n[2/2] レビュー収集...")
    rows = []
    seen_text = set()  # 完全重複の除去
    last_saved = 0

    for i, (app_id, name) in enumerate(candidates):
        if len(rows) >= args.target:
            break
        try:
            reviews = get_steam_reviews(
                app_id=app_id, language='english', review_type='all', num=args.per_game
            )
        except Exception as e:
            print(f"  ❌ {name} ({app_id}) 取得エラー: {e} → スキップ")
            continue

        added = 0
        for r in reviews:
            text = r.get('review_text', '')
            if not text or text in seen_text:
                continue
            seen_text.add(text)
            rows.append({'review_text': text, 'game_id': app_id})
            added += 1

        print(f"  [{i+1}/{len(candidates)}] {name}: +{added}  (累計 {len(rows):,}/{args.target:,})")

        # 途中保存（クラッシュ対策）
        if len(rows) - last_saved >= args.save_every:
            save_corpus(rows, args.output)
            last_saved = len(rows)
            print(f"    💾 途中保存: {len(rows):,}件")

    # 4. 最終保存
    save_corpus(rows, args.output)
    print("\n" + "=" * 60)
    print("✅ 収集完了")
    print("=" * 60)
    print(f"  総レビュー: {len(rows):,}件（重複除去後）")
    print(f"  使用ゲーム数: {len(set(r['game_id'] for r in rows))}本")
    print(f"  保存先: {args.output}")


if __name__ == '__main__':
    main()
