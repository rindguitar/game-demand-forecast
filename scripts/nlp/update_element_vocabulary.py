"""
①ゲーム要素の語彙を、母集団のSteamタグから作り直す

①はかつて「どの語彙にも当たらなければ①」という残余だった。語彙の穴がすべて
需要スコアの対象に落ちるため、①にも証拠を持たせる必要がある（→ Issue #45）。

その語彙を人が書き尽くすのは続かないので、**Steamが整備しているタグをそのまま使う**。
タグは収集時に自動で取れるため、ロスターを広げれば語彙も一緒に広がる。

①の基準は「遊ぶ前に分かる、ゲームの中身」。ゲームの中身を表さないタグ（開発規模・
販売形態・シリーズ等）と、遊んだ結果の感想（Addictive 等）は①にしない（→ configs/steam_tags.txt）。
`Multiplayer` や `Co-op` は残す。企画で「協力プレイを入れるか」を決められる要素だから。

使い方:
    docker compose exec dev python scripts/nlp/update_element_vocabulary.py
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.nlp.tag_semantics import element_tags  # noqa: E402

SECTION = 'element'
HEADER = """# ①ゲーム要素の語彙（自動生成・手で編集しない）
# scripts/nlp/update_element_vocabulary.py が母集団のSteamタグから作る。
# 除いているのは configs/steam_tags.txt で [not_content] [impression] と判定したタグ。"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pool', default='data/timeseries/pool_cache.json')
    parser.add_argument('--categories', default='configs/topic_categories.txt')
    parser.add_argument('--dry-run', action='store_true', help='書き換えずに結果だけ見る')
    return parser.parse_args()


def collect_tags(pool_path: str, judgments: Optional[Dict[str, str]] = None) -> list:
    """母集団のタグを集める（中身を表さないもの・遊んだ結果の感想は除く）

    判定は configs/steam_tags.txt に1つだけ置く（tag_semantics.element_tags が読む）。
    生成側と照合側で判定がずれると、`free` が Free to Play に0.66で①に入ってしまう。
    """
    with open(pool_path, encoding='utf-8') as f:
        pool = {k: v for k, v in json.load(f).items() if k != '__order__'}
    return element_tags(pool, judgments)


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
    print('  除いた基準: configs/steam_tags.txt の [not_content] [impression]')
    print(f'  例: {", ".join(tags[:12])}')
    if args.dry_run:
        print('\n--dry-run のため書き換えない')
        return
    with open(args.categories, 'w', encoding='utf-8') as f:
        f.write(updated)
    print(f'\n✅ 更新: {args.categories}')


if __name__ == '__main__':
    main()
