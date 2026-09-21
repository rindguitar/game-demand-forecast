"""
①ゲーム要素の語彙を母集団のタグから作るスクリプトのテスト

語彙を人が書き尽くすのは続かないので、Steamが整備しているタグを流用する。
除くのは「ゲームの中身を表していない」タグだけ。
"""

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/nlp'))

from update_element_vocabulary import collect_tags, replace_section  # noqa: E402


def _pool(tmp_path, games):
    path = tmp_path / 'pool.json'
    path.write_text(json.dumps(dict(games, __order__=[])), encoding='utf-8')
    return str(path)


def test_collect_tags_gathers_across_games(tmp_path):
    """母集団の全ゲームからタグを集めて重複を畳む"""
    path = _pool(tmp_path, {'1': {'tags': ['Roguelike', 'Horror']},
                            '2': {'tags': ['Horror', 'Sandbox']}})
    assert collect_tags(path) == ['Horror', 'Roguelike', 'Sandbox']


def test_collect_tags_drops_non_elements(tmp_path):
    """開発規模・販売形態・課金モデルのタグは①ではないので除く"""
    path = _pool(tmp_path, {'1': {'tags': ['Indie', 'Early Access', 'Free to Play', 'Horror']}})
    assert collect_tags(path) == ['Horror']


def test_collect_tags_keeps_how_you_play(tmp_path):
    """「誰と遊ぶか」は残す（企画で決められる要素なので）

    似ているかの判定では除外するが（TAG_NOISE）、①の語彙としては有効。
    """
    path = _pool(tmp_path, {'1': {'tags': ['Co-op', 'PvP', 'Multiplayer']}})
    assert collect_tags(path) == ['Co-op', 'Multiplayer', 'PvP']


def test_collect_tags_tolerates_games_without_tags(tmp_path):
    """タグを持たないゲームがあっても落ちない（母集団の143本が該当）"""
    path = _pool(tmp_path, {'1': {'tags': ['Horror']}, '2': {}, '3': {'tags': []}})
    assert collect_tags(path) == ['Horror']


def test_replace_section_appends_when_absent():
    """[element] が無ければ末尾に足す"""
    text = '[quality]\nbug\n'
    out = replace_section(text, ['Horror'])
    assert '[quality]\nbug\n' in out
    assert '[element]\nhorror\n' in out


def test_replace_section_overwrites_existing():
    """既にある [element] は丸ごと入れ替える（古い語を残さない）"""
    text = '[quality]\nbug\n\n[element]\nold_tag\nanother_old\n'
    out = replace_section(text, ['Horror'])
    assert 'old_tag' not in out
    assert 'another_old' not in out
    assert '[element]\nhorror\n' in out


def test_replace_section_does_not_touch_other_sections():
    """他の分類の語彙は変えない"""
    text = '[quality]\nbug\ncrash\n\n[element]\nold\n\n[business]\nprice\n'
    out = replace_section(text, ['Horror'])
    assert 'bug' in out and 'crash' in out and 'price' in out


def test_replace_section_lowercases_tags():
    """語彙は小文字で保存する（照合側が小文字で比べるため）"""
    assert 'souls-like' in replace_section('[element]\n', ['Souls-like'])
