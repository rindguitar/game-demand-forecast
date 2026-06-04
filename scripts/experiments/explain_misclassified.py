"""
誤分類サンプルのモデル解釈性分析

transformers-interpret（Layer Integrated Gradients）を使い、
誤分類CSV（--input）の各レビューに対してトークンごとの寄与度を算出する。

入力: 誤分類CSV（--input・analyze_misclassified.pyの出力）＋ モデル（--model）
出力: <出力先>/ 配下に
    ├── token_scores.csv    詳細：review_id × token × score
    ├── top_words.csv       集計：各レビューの上位5トークン
    └── summary.json        モデル全体の傾向：誤判定を引き起こす単語TOP
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers_interpret import SequenceClassificationExplainer

from src.nlp.model import load_model


class HuggingFaceCompatibleWrapper(nn.Module):
    """
    SentimentClassifierをHuggingFace互換にラップ

    transformers-interpretが内部で参照する属性をすべて満たす:
    - model.config（model_type, id2label, label2id）
    - model.base_model_prefix（base model属性名）
    - model.distilbert（base_model_prefixが指すDistilBERT本体）
    - model.get_input_embeddings()
    - forward() が SequenceClassifierOutput を返す
    """

    base_model_prefix = "distilbert"

    def __init__(self, sentiment_classifier: nn.Module):
        super().__init__()
        self.inner = sentiment_classifier
        # base_model_prefix が指す属性を露出（getattr(model, 'distilbert')で取れるように）
        self.distilbert = self.inner.bert
        # configを設定
        self.config = self.inner.bert.config
        self.config.id2label = {0: 'NEGATIVE', 1: 'POSITIVE'}
        self.config.label2id = {'NEGATIVE': 0, 'POSITIVE': 1}

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        logits = self.inner(input_ids=input_ids, attention_mask=attention_mask)
        return SequenceClassifierOutput(logits=logits)

    def get_input_embeddings(self):
        """transformers-interpretが内部で呼ぶ"""
        return self.distilbert.get_input_embeddings()

    @property
    def device(self):
        """HuggingFaceモデルが持つdeviceプロパティを再現"""
        return next(self.parameters()).device


def parse_args():
    parser = argparse.ArgumentParser(description='誤分類サンプルの解釈性分析')
    parser.add_argument('--input', default='data/experiments/ood_benchmark/misclassified_best_model.csv',
                        help='誤分類CSV（analyze_misclassified.pyの出力）')
    parser.add_argument('--model', default='models/best_model', help='モデルディレクトリ')
    parser.add_argument('--output-dir', default=None,
                        help='出力先（未指定なら入力と同ディレクトリのexplanations/）')
    parser.add_argument('--limit', type=int, default=None,
                        help='処理件数の上限（デフォルト全件）')
    parser.add_argument('--confidence-min', type=float, default=None,
                        help='confidence最小値でフィルタ（例: 0.9で高信頼度誤分類のみ）')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                        help='何件処理ごとに途中保存するか')
    parser.add_argument('--resume', action='store_true',
                        help='既存のtoken_scores.csvを読み込み、未処理分のみ続行')
    return parser.parse_args()


def aggregate_top_words(
    token_scores_df: pd.DataFrame,
    misclassified_df: pd.DataFrame,
    top_n: int = 5
) -> pd.DataFrame:
    """
    各レビューの寄与度TOP N単語を抽出（P方向・N方向の両方）

    出力カラム:
        review_id, review_text, actual, predicted, confidence, error_type,
        pos1_token, pos1_score, ..., pos5_token, pos5_score,
        neg1_token, neg1_score, ..., neg5_token, neg5_score
    """
    # 特殊トークン（[CLS]/[SEP]等）を除外
    df = token_scores_df[~token_scores_df['token'].str.startswith('[')].copy()

    rows = []
    for review_id, grp in df.groupby('review_id'):
        row = {'review_id': review_id}

        # P方向（正の寄与）TOP N
        pos_tokens = grp[grp['score'] > 0].nlargest(top_n, 'score')
        for i, (_, t) in enumerate(pos_tokens.iterrows(), start=1):
            row[f'pos{i}_token'] = t['token']
            row[f'pos{i}_score'] = round(t['score'], 4)
        # 足りない場合は空欄で埋める
        for i in range(len(pos_tokens) + 1, top_n + 1):
            row[f'pos{i}_token'] = ''
            row[f'pos{i}_score'] = ''

        # N方向（負の寄与）TOP N
        neg_tokens = grp[grp['score'] < 0].nsmallest(top_n, 'score')
        for i, (_, t) in enumerate(neg_tokens.iterrows(), start=1):
            row[f'neg{i}_token'] = t['token']
            row[f'neg{i}_score'] = round(t['score'], 4)
        for i in range(len(neg_tokens) + 1, top_n + 1):
            row[f'neg{i}_token'] = ''
            row[f'neg{i}_score'] = ''

        rows.append(row)

    top_df = pd.DataFrame(rows)

    # misclassified情報を結合
    meta_cols = ['review_id', 'review_text', 'actual',
                 'predicted', 'confidence', 'error_type']
    merged = misclassified_df[meta_cols].merge(top_df, on='review_id', how='inner')

    # カラム順を整理: メタ情報 → P方向 → N方向
    pos_cols = [f'pos{i}_{x}' for i in range(1, top_n + 1) for x in ['token', 'score']]
    neg_cols = [f'neg{i}_{x}' for i in range(1, top_n + 1) for x in ['token', 'score']]
    result = merged[meta_cols + pos_cols + neg_cols]

    # confidence降順でソート（高信頼度誤分類が上に来る）
    return result.sort_values('confidence', ascending=False).reset_index(drop=True)


# 英語ストップワード（頻出だが意味希薄なため集計から除外）
STOPWORDS = {
    'i', 'a', 'an', 'the', 'and', 'or', 'but', 'if', 'so', 'to', 'of', 'in', 'on',
    'at', 'for', 'with', 'as', 'by', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
    'could', 'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those',
    'my', 'your', 'his', 'her', 'its', 'our', 'their', 'me', 'you', 'him', 'us',
    'them', 'we', 'he', 'she', 'it', 'they', 'what', 'which', 'who', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'some', 'any', 'no', 'not',
    'only', 'own', 'same', 'than', 'then', 'too', 'very', 'just', 'one', 'two',
    'there', 'here', 'where',
}

# 短縮形の断片（don't, you've, it's等がtokenizerで分割された残り）
CONTRACTION_FRAGMENTS = {
    # アポストロフィの後ろ部分
    's', 't', 'd', 'm', 'll', 've', 're',
    # アポストロフィの前部分（"...n't"系）
    'don', 'won', 'isn', 'wasn', 'aren', 'weren',
    'doesn', 'didn', 'couldn', 'wouldn', 'shouldn',
    'hasn', 'haven', 'hadn', 'mustn', 'mightn', 'needn',
    'ain', 'shan',
}

# 集計対象とする最小トークン長（短い断片を除外）
MIN_TOKEN_LENGTH = 3


def _is_meaningful_token(token: str) -> bool:
    """集計対象として意味のあるトークンか判定"""
    if not isinstance(token, str):
        return False
    # 特殊トークン除外
    if token.startswith('['):
        return False
    # サブワード断片除外（##で始まる）
    if token.startswith('##'):
        return False
    # 記号のみ除外
    if not any(c.isalpha() for c in token):
        return False
    # 最小長フィルタ（s, t, ve, ll, re, m, d等の断片除外）
    if len(token) < MIN_TOKEN_LENGTH:
        return False
    # ストップワード除外
    if token.lower() in STOPWORDS:
        return False
    # 短縮形断片除外
    if token.lower() in CONTRACTION_FRAGMENTS:
        return False
    return True


def build_summary(
    token_scores_df: pd.DataFrame,
    misclassified_df: pd.DataFrame,
    min_occurrences: int = 5,
    top_n: int = 20,
) -> dict:
    """モデル全体の傾向サマリーを作成

    Args:
        min_occurrences: 集計に含める最小出現回数（ノイズ除去）
        top_n: 各カテゴリで上位何件を返すか
    """
    # error_type を結合
    df = token_scores_df.merge(
        misclassified_df[['review_id', 'error_type']],
        on='review_id', how='left'
    )

    # 意味のあるトークンのみに絞る（特殊トークン・記号・サブワード・ストップワード除外）
    df = df[df['token'].apply(_is_meaningful_token)]

    summary = {
        'total_reviews_analyzed': int(token_scores_df['review_id'].nunique()),
        'total_tokens_after_filter': int(len(df)),
        'filter_criteria': {
            'min_occurrences': min_occurrences,
            'stopwords_excluded': True,
            'subwords_excluded': True,
            'special_tokens_excluded': True,
        }
    }

    def _aggregate(sub_df: pd.DataFrame, ascending: bool) -> list:
        """token単位で集計し、mean_score基準で並べる（最小出現回数フィルタ付き）"""
        agg = sub_df.groupby('token')['score'].agg(['sum', 'mean', 'count'])
        agg = agg[agg['count'] >= min_occurrences]
        agg = agg.sort_values('mean', ascending=ascending).head(top_n)
        return [
            {
                'token': str(idx),
                'mean_score': round(r['mean'], 4),
                'total_score': round(r['sum'], 4),
                'occurrences': int(r['count']),
            }
            for idx, r in agg.iterrows()
        ]

    # FP（NをPと誤認）でポジ方向に寄与した単語
    fp_pos = df[(df['error_type'] == 'FP') & (df['score'] > 0)]
    summary['fp_top_positive_contributors'] = _aggregate(fp_pos, ascending=False)

    # FN（PをNと誤認）でネガ方向に寄与した単語
    fn_neg = df[(df['error_type'] == 'FN') & (df['score'] < 0)]
    summary['fn_top_negative_contributors'] = _aggregate(fn_neg, ascending=True)

    return summary


def main():
    args = parse_args()

    input_path = args.input
    model_dir = args.model
    output_dir = args.output_dir or os.path.join(os.path.dirname(input_path), 'explanations')
    token_scores_path = os.path.join(output_dir, 'token_scores.csv')
    top_words_path = os.path.join(output_dir, 'top_words.csv')
    summary_path = os.path.join(output_dir, 'summary.json')

    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print('誤分類サンプルのモデル解釈性分析（Layer Integrated Gradients）')
    print('=' * 70)

    # 1. データ読み込み
    df = pd.read_csv(input_path).reset_index(drop=True)
    df['review_id'] = df.index

    # フィルタ適用
    if args.confidence_min is not None:
        df = df[df['confidence'] >= args.confidence_min].copy()
    if args.limit is not None:
        df = df.head(args.limit).copy()

    print(f'\n[1/4] 入力読み込み: {len(df)}件')

    # 既存処理分（resume対応）
    done_ids = set()
    if args.resume and os.path.exists(token_scores_path):
        existing = pd.read_csv(token_scores_path)
        done_ids = set(existing['review_id'].unique())
        print(f'   既存処理済み: {len(done_ids)}件 (--resume指定)')
    elif not args.resume and os.path.exists(token_scores_path):
        # --resume指定なしで既存ファイルがある場合は削除（重複追加防止）
        os.remove(token_scores_path)
        print(f'   既存のtoken_scores.csvを削除（--resume未指定のため）')

    # 2. モデル読み込み
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n[2/4] モデル読み込み (device={device})')
    model, tokenizer = load_model(model_dir, device=device)
    model.eval()

    # transformers-interpretはHuggingFace互換のmodel.configを期待するためラップ
    wrapped_model = HuggingFaceCompatibleWrapper(model).to(device)
    wrapped_model.eval()

    explainer = SequenceClassificationExplainer(
        model=wrapped_model,
        tokenizer=tokenizer,
    )

    # 3. 解釈性分析
    print(f'\n[3/4] 解釈性分析 (Layer Integrated Gradients)')

    target_class_idx = {0: 0, 1: 1}  # 0=Negative, 1=Positive
    rows_buffer = []
    processed_count = 0

    # 新規ファイルなら最初にヘッダー出力、--resumeで既存に追記する場合はヘッダーなし
    write_header = not os.path.exists(token_scores_path)

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Explaining'):
        review_id = int(row['review_id'])
        if review_id in done_ids:
            continue

        text = str(row['review_text'])
        # モデルが予測したクラスに対する寄与度を計算
        predicted_class = 1 if row['predicted'] == 'P' else 0

        try:
            attributions = explainer(
                text=text,
                class_name=str(predicted_class),
            )
        except Exception as e:
            print(f'   ⚠ review_id={review_id} スキップ: {e}')
            continue

        for token, score in attributions:
            rows_buffer.append({
                'review_id': review_id,
                'token': token,
                'score': float(score),
            })

        processed_count += 1

        # チェックポイント保存
        if processed_count % args.checkpoint_interval == 0:
            tmp_df = pd.DataFrame(rows_buffer)
            tmp_df.to_csv(token_scores_path, mode='a',
                          header=write_header, index=False)
            write_header = False
            rows_buffer.clear()

    # 残り分を保存
    if rows_buffer:
        tmp_df = pd.DataFrame(rows_buffer)
        tmp_df.to_csv(token_scores_path, mode='a',
                      header=write_header, index=False)

    print(f'\n   ✓ token_scores.csv 保存: {token_scores_path}')

    # 4. 集計・サマリー作成
    print(f'\n[4/4] 集計・サマリー作成')
    token_scores_df = pd.read_csv(token_scores_path)

    # misclassified情報の再読み込み（review_text等を結合用に使う）
    misclassified_df = pd.read_csv(input_path).reset_index(drop=True)
    misclassified_df['review_id'] = misclassified_df.index

    # 上位5単語（P方向・N方向それぞれ + メタ情報付き）
    top_words_df = aggregate_top_words(token_scores_df, misclassified_df, top_n=5)
    # 出力からreview_id削除（confidence順なので不要）
    top_words_df = top_words_df.drop(columns=['review_id'])
    top_words_df.to_csv(top_words_path, index=False)
    print(f'   ✓ top_words.csv 保存: {top_words_path}')

    # token_scores.csv を最終整形（review_text/confidence/error_type追加・ソート・review_id削除）
    enriched = token_scores_df.merge(
        misclassified_df[['review_id', 'review_text', 'confidence', 'error_type']],
        on='review_id', how='left'
    )
    enriched = enriched.sort_values(
        ['confidence', 'score'], ascending=[False, False]
    ).reset_index(drop=True)
    final_token_scores = enriched[
        ['review_text', 'confidence', 'error_type', 'token', 'score']
    ]
    final_token_scores.to_csv(token_scores_path, index=False)
    print(f'   ✓ token_scores.csv 最終整形: {token_scores_path}')

    # サマリー
    summary = build_summary(token_scores_df, misclassified_df)

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'   ✓ summary.json 保存: {summary_path}')

    # コンソール表示
    print(f'\n【サマリー】')
    print(f'  分析件数: {summary["total_reviews_analyzed"]}件')
    print(f'  フィルタ後トークン数: {summary["total_tokens_after_filter"]}件')
    print(f'  フィルタ条件: 最小出現{summary["filter_criteria"]["min_occurrences"]}回, '
          f'ストップワード/サブワード/特殊トークン除外')

    print(f'\n【FP（NをPと誤認）でポジ方向に寄与した単語 TOP10（mean_score順）】')
    for item in summary['fp_top_positive_contributors'][:10]:
        print(f"  {item['token']:15} | mean={item['mean_score']:>6.3f} | "
              f"sum={item['total_score']:>7.3f} | 出現={item['occurrences']}回")

    print(f'\n【FN（PをNと誤認）でネガ方向に寄与した単語 TOP10（mean_score順）】')
    for item in summary['fn_top_negative_contributors'][:10]:
        print(f"  {item['token']:15} | mean={item['mean_score']:>6.3f} | "
              f"sum={item['total_score']:>7.3f} | 出現={item['occurrences']}回")

    print('\n' + '=' * 70)
    print('解釈性分析完了')
    print('=' * 70)


if __name__ == '__main__':
    main()
