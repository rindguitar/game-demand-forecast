"""
トピックモデルの学習用サンプルを作るスクリプト

複数のレビューCSV（例: 64本と85本の時系列レビュー）から、ゲームごとに上限をかけて
学習用のサンプルを1つのCSVにまとめる。

なぜ別スクリプトか: 2つ合わせて約2GBあり、つないでから extract_topics.py に
渡すとメモリが足りなくなる。1ファイルずつ読んでサンプルにしてから、つなぐ。
学習用サンプルが決めるのは「どんなトピックがあるか（顔ぶれ）」だけなので、
量の多いゲームに偏らないよう、ゲームごとの上限（--per-game）をかける
（extract_topics.py の sample_for_fit と同じ規則）。

処理の流れ:
    1. CSV を1ファイルずつ読む（必要な列だけ）→ 2. review_text が空の行を落とす
    → 3. ゲームごとに上限をかけて抜く → 4. 全部つなぐ
    → 5. 総数が --total を超えていれば無作為に切る → 6. CSV に保存

使い方は --help を参照。
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from extract_topics import sample_for_fit

USE_COLUMNS = ['game_id', 'game_name', 'review_text']


def make_fit_sample(inputs, per_game: int, total: int, seed: int) -> pd.DataFrame:
    """
    複数CSVから学習用サンプルを作る

    1ファイルずつ「読む → 空の本文を落とす → ゲームごとの上限」をかけてからつなぎ、
    最後に total を超えていれば無作為に切る。
    """
    parts = []
    for path in inputs:
        df = pd.read_csv(path, usecols=lambda c: c in USE_COLUMNS)
        df = df.dropna(subset=['review_text'])
        df = df[df['review_text'].astype(str).str.strip() != '']
        part = sample_for_fit(df, per_game=per_game, total=0, seed=seed)
        print(f"   {path}: 読み込み {len(df):,}件 → サンプル {len(part):,}件 "
              f"/ {part['game_name'].nunique()}本")
        parts.append(part)

    sample = pd.concat(parts, ignore_index=True)
    if total and len(sample) > total:
        sample = sample.sample(n=total, random_state=seed).reset_index(drop=True)
    return sample


def main():
    """学習用サンプル作成のメイン実行関数"""
    parser = argparse.ArgumentParser(description='トピックモデルの学習用サンプルを作る')
    parser.add_argument('--inputs', nargs='+', required=True, help='レビューCSV（複数可）')
    parser.add_argument('--per-game', type=int, default=5000, help='1本あたりの上限件数')
    parser.add_argument('--total', type=int, default=100000, help='全体の上限件数（0=無制限）')
    parser.add_argument('--seed', type=int, default=42, help='サンプリングのシード')
    parser.add_argument('--output', required=True, help='出力CSV')
    args = parser.parse_args()

    print("学習用サンプルを作る")
    sample = make_fit_sample(args.inputs, args.per_game, args.total, args.seed)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    sample.to_csv(args.output, index=False)

    per = sample['game_name'].value_counts()
    print(f"最終: {len(sample):,}件 / {len(per)}本 / 1本あたり 最多{per.max():,}件・最少{per.min():,}件")
    print(f"保存: {args.output}")


if __name__ == '__main__':
    main()
