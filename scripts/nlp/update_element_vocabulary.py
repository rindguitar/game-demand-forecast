"""
①ゲーム要素の語彙を、母集団のSteamタグから作り直す

①はかつて「どの語彙にも当たらなければ①」という残余だった。語彙の穴がすべて
需要スコアの対象に落ちるため、①にも証拠を持たせる必要がある（→ Issue #45）。

その語彙を人が書き尽くすのは続かないので、**Steamが整備しているタグをそのまま使う**。
タグは収集時に自動で取れるため、ロスターを広げれば語彙も一緒に広がる。

除くのは「ゲームの中身を表していない」タグだけ（開発規模・販売形態・課金モデル）。
`Multiplayer` や `Co-op` は残す。企画で「協力プレイを入れるか」を決められる要素だから。

使い方:
    docker compose exec dev python scripts/nlp/update_element_vocabulary.py
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.nlp.tag_semantics import NOT_ELEMENT_TAGS, element_tags  # noqa: E402

SECTION = 'element'
HEADER = """# ①ゲーム要素の語彙（自動生成・手で編集しない）
# scripts/nlp/update_element_vocabulary.py が母集団のSteamタグから作る。
# 除いているのは開発規模・販売形態・課金モデルのタグだけ（→ tag_semantics.NOT_ELEMENT_TAGS）。"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pool', default='data/timeseries/pool_cache.json')
    parser.add_argument('--categories', default='configs/topic_categories.txt')
    parser.add_argument('--dry-run', action='store_true', help='書き換えずに結果だけ見る')
    return parser.parse_args()


def collect_tags(pool_path: str) -> list:
    """母集団のタグを集める（中身を表さないものは除く）

    除外の定義は src/nlp/tag_semantics.py に置く。意味の照合側と同じ語彙でないと、
    生成側だけ除いても照合側で①に入ってしまう（実測: `free` が Free to Play に0.66）。
    """
    with open(pool_path, encoding='utf-8') as f:
        pool = {k: v for k, v in json.load(f).items() if k != '__order__'}
    return element_tags(pool)


def replace_section(text: str, tags: list) -> str:
    """[element] セクションを丸ごと入れ替える（無ければ末尾に足す）"""
    block = f"{HEADER}\n[{SECTION}]\n" + '\n'.join(t.lower() for t in tags) + '\n'
    pattern = re.compile(rf'(?:^#[^\n]*\n)*^\[{SECTION}\]\n(?:[^\[\n][^\n]*\n|\n)*',
                         re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(block, text, count=1)
    return text.rstrip('\n') + '\n\n' + block


def main():
    args = parse_args()
    tags = collect_tags(args.pool)
    with open(args.categories, encoding='utf-8') as f:
        text = f.read()
    updated = replace_section(text, tags)

    print(f'①ゲーム要素の語彙: {len(tags)}語（{args.pool} のタグから）')
    print(f'  除いたもの: {", ".join(sorted(NOT_ELEMENT_TAGS))}')
    print(f'  例: {", ".join(tags[:12])}')
    if args.dry_run:
        print('\n--dry-run のため書き換えない')
        return
    with open(args.categories, 'w', encoding='utf-8') as f:
        f.write(updated)
    print(f'\n✅ 更新: {args.categories}')


if __name__ == '__main__':
    main()
