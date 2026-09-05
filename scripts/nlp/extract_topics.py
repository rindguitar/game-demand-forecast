"""
トピック抽出実行スクリプト（本番用）

レビューCSV（--input）からBERTopicでトピックを抽出し、結果をCSVに保存する。

件数が多い場合は、**一部で学習（fit）→ 全件に割り当て（transform）** に分ける。
次元削減とクラスタリングが件数に弱く、数十万件を一度に投げるとメモリが足りなく
なるため（詳細は Wiki「トピック抽出」「次元削減とクラスタリング」）。

学習用サンプルが決めるのは**どんなトピックがあるか（顔ぶれ）だけ**で、各トピックの
件数は全件から数える。そのためサンプルはゲームごとに上限を設けて取る
（--sample-per-game）。単純にランダムだと、量の多いゲームの話題がトピックの
顔ぶれを占めてしまう（実データでは1本が全体の19.7%）。

使い方は --help を参照。
"""

import sys
import os
import time
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
from src.nlp.topic import (
    filter_english_reviews,
    remove_game_names,
    load_proper_nouns,
    create_topic_model,
    extract_topics,
    assign_topics,
    save_topic_model,
    load_topic_model,
    print_topic_summary,
    get_topic_info,
    get_topic_words
)


def sample_for_fit(df: pd.DataFrame, per_game: int, total: int, seed: int) -> pd.DataFrame:
    """
    学習用のサンプルを作る（ゲームごとに上限を設けてから、必要なら総数で切る）

    シャッフルしてから各ゲームの先頭を取ることで、ゲーム内では偏らずに抜き出す。
    """
    sample = df.sample(frac=1.0, random_state=seed)
    if per_game and 'game_name' in sample.columns:
        sample = sample.groupby('game_name', group_keys=False).head(per_game)
    if total and len(sample) > total:
        sample = sample.sample(n=total, random_state=seed)
    return sample


def main():
    """トピック抽出のメイン実行関数"""

    parser = argparse.ArgumentParser(description='BERTopicでトピック抽出')
    parser.add_argument('--input', default='data/train/reviews_10000.csv',
                        help='レビューCSV（review_text列が必須）')
    parser.add_argument('--output', default=None,
                        help='トピック付与CSV出力先（未指定なら <input>_with_topics.csv）')
    parser.add_argument('--stats-output', default=None,
                        help='トピック統計CSV出力先（未指定なら入力と同ディレクトリのtopic_statistics.csv）')
    parser.add_argument('--min-topic-size', type=int, default=20,
                        help='トピックと認める最小レビュー数。大きいほど粗く、小さいほど細かくなる')
    parser.add_argument('--sample-per-game', type=int, default=0,
                        help='学習用サンプルの1ゲームあたり上限（0=制限なし）。'
                             '量の多いゲームがトピックの顔ぶれを占めるのを防ぐ')
    parser.add_argument('--fit-sample-size', type=int, default=0,
                        help='学習用サンプルの総数上限（0=制限なし）')
    parser.add_argument('--limit', type=int, default=0,
                        help='処理対象そのものをこの件数に絞る（試走用・ランダム抽出）')
    parser.add_argument('--remove-all-game-names', action='store_true',
                        help='全ゲームの名前を全レビューから除去する。既定は自分の名前だけで、'
                             'その場合は他ゲームへの言及が残りトピックがゲーム専用になる')
    parser.add_argument('--proper-nouns', default='configs/proper_nouns.txt',
                        help='追加で除去する固有名詞のファイル（開発元名など）')
    parser.add_argument('--skip-english-filter', action='store_true',
                        help='英語フィルタを掛けない。収集時に通過済みのデータでは二重になる')
    parser.add_argument('--model-output', default=None,
                        help='学習済みモデルの保存先ディレクトリ')
    parser.add_argument('--model-input', default=None,
                        help='学習済みモデルを読み込んで再学習しない')
    parser.add_argument('--seed', type=int, default=42, help='サンプリングのシード')
    args = parser.parse_args()

    input_path = args.input
    stem, ext = os.path.splitext(input_path)
    output_path = args.output or f'{stem}_with_topics{ext}'
    stats_output_path = args.stats_output or os.path.join(
        os.path.dirname(input_path), 'topic_statistics.csv')

    print("=" * 80)
    print("🔍 トピック抽出実行")
    print("=" * 80)
    t0 = time.time()

    # 1. データ読み込み
    print("\n[1/6] データ読み込み")
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['review_text'])
    print(f"   ✓ 読み込み完了: {len(df):,}件（{input_path}）")
    if args.limit and len(df) > args.limit:
        df = df.sample(n=args.limit, random_state=args.seed)
        print(f"   ✓ --limit により {len(df):,}件に絞った（試走）")

    # 2. 英語レビューのみフィルタリング
    print("\n[2/6] 英語レビューフィルタリング")
    if args.skip_english_filter:
        print("   ✓ --skip-english-filter のためスキップ（収集時に通過済み）")
    else:
        original_count = len(df)
        df = filter_english_reviews(df, text_column='review_text')
        filtered = original_count - len(df)
        print(f"   ✓ 英語レビュー: {len(df):,}件 / 除外: {filtered:,}件 "
              f"({filtered / original_count * 100:.1f}%)")

    # 3. 固有名詞の除去（ゲーム名でトピックが割れるのを防ぐ）
    print("\n[3/6] 固有名詞の除去")
    extra = load_proper_nouns(args.proper_nouns) if args.remove_all_game_names else []
    if extra:
        print(f"   ✓ 追加の固有名詞: {len(extra)}語（{args.proper_nouns}）")
    df = remove_game_names(df, text_column='review_text', game_name_column='game_name',
                           all_games=args.remove_all_game_names, extra_words=extra)
    texts = df['review_text'].tolist()

    # 4. 学習用サンプルの決定
    print("\n[4/6] 学習用サンプル")
    fit_df = sample_for_fit(df, args.sample_per_game, args.fit_sample_size, args.seed)
    fit_texts = fit_df['review_text'].tolist()
    split_fit = len(fit_texts) < len(texts)
    print(f"   ✓ 学習に使う: {len(fit_texts):,}件 / 割り当て対象: {len(texts):,}件")
    if 'game_name' in fit_df.columns:
        per = fit_df['game_name'].value_counts()
        print(f"   ✓ ゲーム別: {len(per)}本 / 最多{per.max():,}件 / 最少{per.min():,}件")

    # 5. 学習（または保存済みモデルの読み込み）
    print("\n[5/6] トピックモデルの学習")
    t_fit = time.time()
    if args.model_input:
        topic_model = load_topic_model(args.model_input)
        print(f"   ✓ 学習済みモデルを読み込み: {args.model_input}")
    else:
        topic_model = create_topic_model(
            min_topic_size=args.min_topic_size,
            embedding_model_name='all-MiniLM-L6-v2',
            ngram_range=(1, 2),
            min_df=2,
            verbose=True
        )
        print(f"   ✓ min_topic_size: {args.min_topic_size}")
        print("   ⏳ 学習中（次元削減とクラスタリングに時間がかかる）...")
        topic_model, fit_topics, _ = extract_topics(
            texts=fit_texts, topic_model=topic_model, verbose=True)
        if args.model_output:
            save_topic_model(topic_model, args.model_output)
            print(f"   ✓ モデルを保存: {args.model_output}")
    fit_sec = time.time() - t_fit

    # 6. 全件にトピックを割り当てる
    print("\n[6/6] 全件へのトピック割り当て")
    t_tr = time.time()
    if split_fit or args.model_input:
        topics, probabilities = assign_topics(topic_model, texts, verbose=True)
    else:
        topics, probabilities = fit_topics, None
        print("   ✓ 学習と対象が同じなので、学習時の結果をそのまま使う")
    tr_sec = time.time() - t_tr

    num_topics = len(set(topics) - {-1})
    outlier_count = topics.count(-1)
    print(f"\n   ✓ 抽出トピック数: {num_topics}")
    print(f"   ✓ Outlier: {outlier_count:,}件 ({outlier_count / len(topics) * 100:.1f}%)")
    print(f"   ✓ 所要時間: 学習 {fit_sec:.0f}秒 / 割り当て {tr_sec:.0f}秒")

    # 結果サマリー
    print_topic_summary(
        topic_model=topic_model, topics=topics, texts=texts,
        max_topics=20, top_n_words=10, sample_reviews=3
    )

    # 保存
    print("\n" + "=" * 80)
    print("💾 結果保存")
    print("=" * 80)

    df['topic_id'] = topics
    if probabilities is not None:
        df['topic_probability'] = probabilities

    topic_info_df = get_topic_info(topic_model)
    topic_name_map = dict(zip(topic_info_df['Topic'], topic_info_df['Name']))
    df['topic_name'] = df['topic_id'].map(topic_name_map)

    topic_words_map = {}
    for topic_id in sorted(set(topics) - {-1}):
        words = get_topic_words(topic_model, topic_id, top_n=5)
        topic_words_map[topic_id] = ", ".join([w for w, _ in words]) if words else ""
    topic_words_map[-1] = "Outlier"
    df['topic_keywords'] = df['topic_id'].map(topic_words_map)

    df.to_csv(output_path, index=False)
    print(f"\n   ✓ レビュー+トピック結果: {output_path}")

    # トピック統計
    counts = pd.Series(topics).value_counts()
    topic_stats = [{
        'topic_id': tid,
        'topic_name': topic_name_map.get(tid, 'Unknown'),
        'keywords': topic_words_map.get(tid, ''),
        'count': int(n),
        'percentage': n / len(topics) * 100,
    } for tid, n in counts.items()]
    pd.DataFrame(topic_stats).sort_values('count', ascending=False).to_csv(
        stats_output_path, index=False)
    print(f"   ✓ トピック統計: {stats_output_path}")

    print("\n" + "=" * 80)
    print("✅ トピック抽出完了")
    print("=" * 80)
    print(f"   - 処理レビュー数: {len(texts):,}件（学習 {len(fit_texts):,}件）")
    print(f"   - 抽出トピック数: {num_topics}")
    print(f"   - Outlier: {outlier_count:,}件 ({outlier_count / len(topics) * 100:.1f}%)")
    print(f"   - 全体の所要時間: {time.time() - t0:.0f}秒")
    print()


if __name__ == '__main__':
    main()
