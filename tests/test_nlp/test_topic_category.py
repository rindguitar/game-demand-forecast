"""
トピック分類モジュールのテスト
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pytest
from src.nlp.topic_category import (
    ELEMENT,
    UNCLASSIFIED,
    QUALITY,
    BUSINESS,
    CONTENTLESS,
    PROPERNOUN,
    AMBIGUOUS,
    classify_topic,
    classify_topics,
    format_hits,
    load_category_words,
)


@pytest.fixture
def words():
    """本番の語彙ではなく、テスト用の最小の語彙"""
    return {
        QUALITY: ['crash', 'crashes', 'bug', 'performance'],
        BUSINESS: ['price', 'dlc', 'dlcs', 'free game'],
        CONTENTLESS: ['best game', 'hours', 'recommend'],
        PROPERNOUN: ['bungie', 'team cherry'],
        ELEMENT: [],
    }


def test_classify_topic_unmatched_is_unclassified(words):
    """どの語彙にも当たらないトピックは未分類

    かつては①ゲーム要素だった。残余を需要スコアの対象にすると、語彙の穴が
    そのまま混入する（実測: 64本で boobs / braindead / money が①に入った）。
    """
    category, hits = classify_topic('hunting, animals, hunting game, animal, hunt', words)
    assert category == UNCLASSIFIED
    assert all(not v for v in hits.values())


def test_classify_topic_quality(words):
    """不具合の語に当たれば②品質・運営"""
    category, hits = classify_topic('crashes, crashing, crash, game crashes', words)
    assert category == QUALITY
    assert 'crash' in hits[QUALITY]


def test_classify_topic_business(words):
    """価格・DLCの語に当たれば③ビジネス条件"""
    assert classify_topic('dlcs, base game, sale, game dlcs, base', words)[0] == BUSINESS


def test_classify_topic_contentless(words):
    """賞賛・プレイ時間の語に当たれば中身なし"""
    assert classify_topic('hours, hours game, played hours, game hours', words)[0] == CONTENTLESS


def test_classify_topic_phrase_matches_across_separators(words):
    """語句は空白以外の区切りでも当たる（"free game" が "free, game" に当たる）"""
    category, hits = classify_topic('free, game, best free', words)
    assert category == BUSINESS
    assert 'free game' in hits[BUSINESS]


def test_classify_topic_tie_is_ambiguous(words):
    """複数の分類が同数で当たったら手動送りにする"""
    category, hits = classify_topic('crash, price', words)
    assert category == AMBIGUOUS
    assert hits[QUALITY] and hits[BUSINESS]


def test_classify_topic_majority_wins(words):
    """当たった語が多い分類を採る（同数でなければ曖昧にしない）"""
    assert classify_topic('crash, crashes, bug, price', words)[0] == QUALITY


def test_classify_topic_matches_whole_words_only(words):
    """部分一致では当たらない（"priceless" の中の "price" では当たらない）"""
    assert classify_topic('priceless, artwork', words)[0] == UNCLASSIFIED


def test_classify_topic_plural_must_be_listed(words):
    """複数形は自動では当たらない。語彙に明示的に載せる方針（configs/topic_categories.txt）

    語尾の s を機械的に落とすと "bugs"→"bug" のような正しい場合と一緒に
    "hours"→"hour" のような取りこぼしも作るため、明示列挙にしている。
    """
    assert classify_topic('crashes only', words)[0] == QUALITY   # crashes は語彙にある
    assert classify_topic('bugs only', words)[0] == UNCLASSIFIED  # bugs は語彙に無い


def test_classify_topic_propernoun_wins_over_tie(words):
    """固有名詞は他と同数で当たっても曖昧にせず確定させる

    束ねてはいけないものなので、ambiguous に落として取りこぼすと
    その塊がタグの束に流れ込む。
    """
    assert classify_topic('bungie, price', words)[0] == PROPERNOUN


def test_classify_topic_propernoun_phrase(words):
    """語句の固有名詞も当たる（"cherry" 単体は語彙に入れない方針）"""
    assert classify_topic('team cherry, cherry, thank team', words)[0] == PROPERNOUN
    assert classify_topic('cherry blossom, tree', words)[0] == UNCLASSIFIED


def test_classify_topics_keeps_order(words):
    """一覧はそのままの順序で返る"""
    result = classify_topics([(1, 'crash'), (2, 'hunting'), (3, 'dlc')], words)
    assert [r[0] for r in result] == [1, 2, 3]
    assert [r[2] for r in result] == [QUALITY, UNCLASSIFIED, BUSINESS]


def test_load_category_words(tmp_path):
    """見出しで区切られたファイルを読む。コメントと空行は無視する"""
    path = tmp_path / 'cats.txt'
    path.write_text('# コメント\n\n[quality]\nbug\ncrash\n\n[business]\n# 値段\nprice\n',
                    encoding='utf-8')
    words = load_category_words(str(path))
    assert words[QUALITY] == ['bug', 'crash']
    assert words[BUSINESS] == ['price']
    assert words[CONTENTLESS] == []


def test_load_category_words_missing_file_warns(tmp_path, capsys):
    """ファイルが無いときは黙って空を返さず警告する"""
    words = load_category_words(str(tmp_path / 'none.txt'))
    assert all(v == [] for v in words.values())
    assert '⚠️' in capsys.readouterr().out


def test_load_category_words_ignores_unknown_section(tmp_path):
    """知らない見出しの中身は読み捨てる"""
    path = tmp_path / 'cats.txt'
    path.write_text('[unknown]\nfoo\n[quality]\nbug\n', encoding='utf-8')
    words = load_category_words(str(path))
    assert words[QUALITY] == ['bug']
    assert 'foo' not in sum(words.values(), [])


def test_format_hits_is_readable(words):
    """当たった語の表示は「分類: 語, 語」の形になる"""
    _, hits = classify_topic('crash, bug, price', words)
    text = format_hits(hits)
    assert 'quality:' in text and 'business:' in text


def test_classify_topic_element_needs_evidence(words):
    """①ゲーム要素は語彙に当たって初めて①になる（残余ではない）"""
    words = dict(words, element=['sandbox', 'roguelike'])
    assert classify_topic('roguelike, rogue like, roguelites', words)[0] == ELEMENT
    assert classify_topic('nothing matches here', words)[0] == UNCLASSIFIED


def test_element_and_contentless_tie_goes_to_manual(words):
    """①の証拠と中身なしの証拠が拮抗したら手動送り

    実例: `sandbox, sandbox game, best sandbox` は sandbox（要素）と
    game best（中身なし）が1票ずつで並ぶ。かつては要素側に語彙が無く、
    中身なしの1票だけで中身なしに倒れていた。
    """
    words = dict(words, element=['sandbox'], contentless=list(words['contentless']) + ['game best'])
    assert classify_topic('sandbox, sandbox game, game best', words)[0] == AMBIGUOUS


def test_propernoun_still_wins_over_element(words):
    """固有名詞は①の証拠があっても優先される（束ねる対象から外すため）"""
    words = dict(words, element=['metroidvania'])
    category, _ = classify_topic('metroidvania, team cherry', words)
    assert category == PROPERNOUN


def test_strong_tag_score_beats_word_counting(words):
    """タグとの近さが強ければ、語の数の多数決を飛び越えて①になる

    実例: `sandbox, sandbox game, best sandbox` は `best game` の1票で
    中身なしに落ちていた。タグ Sandbox との近さ 0.78 を強い証拠として扱う。
    """
    category, _ = classify_topic('sandbox, sandbox game, best game', words,
                                 element_score=0.78)
    assert category == ELEMENT


def test_weak_tag_score_only_adds_one_vote(words):
    """弱い近さは1票にしかならない（他に証拠があれば負ける）"""
    # 中身なしが2語当たれば、①の1票では勝てない
    category, _ = classify_topic('best game, hours, something', words, element_score=0.52)
    assert category == CONTENTLESS
    # 拮抗すれば手動送り
    category, _ = classify_topic('best game, something', words, element_score=0.52)
    assert category == AMBIGUOUS


def test_tag_score_below_threshold_is_ignored(words):
    """当てにならない帯の近さは証拠として数えない"""
    category, _ = classify_topic('nothing matches', words, element_score=0.40)
    assert category == UNCLASSIFIED


def test_thresholds_are_configurable(words):
    """閾値は呼び出し側で変えられる（実測で決めるため）"""
    assert classify_topic('nothing', words, element_score=0.55,
                          strong_element=0.50, weak_element=0.45)[0] == ELEMENT
    assert classify_topic('nothing', words, element_score=0.55,
                          strong_element=0.90, weak_element=0.80)[0] == UNCLASSIFIED
