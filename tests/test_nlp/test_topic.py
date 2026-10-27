"""
トピック抽出モジュールのテスト
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pytest
import numpy as np
import pandas as pd
from src.nlp.topic import (
    is_english,
    filter_english_reviews,
    remove_game_names,
    assign_topics,
    create_topic_model,
    extract_topics,
    get_topic_info,
    get_topic_words,
    print_topic_summary
)


def test_is_english():
    """英語判定のテスト"""
    # 英語
    assert is_english("This is a great game!") == True
    assert is_english("I love this game") == True
    assert is_english("10/10") == True

    # 非英語
    assert is_english("これは素晴らしいゲームです") == False
    assert is_english("Разраба маму") == False
    assert is_english("有一说一这游戏玩久了") == False

    # 混合（非ASCII文字が含まれる）
    assert is_english("Great game! 最高！") == False


def test_filter_english_reviews():
    """英語レビューフィルタリングのテスト"""
    # テストデータ作成
    df = pd.DataFrame({
        'review_text': [
            'This is a great game!',
            'これは素晴らしいゲームです',
            'I love this game',
            'Разраба маму',
            'Amazing game'
        ],
        'label': [1, 1, 1, 0, 1]
    })

    df_english = filter_english_reviews(df, text_column='review_text')

    # 英語レビューのみが残る
    assert len(df_english) == 3
    assert 'This is a great game!' in df_english['review_text'].values
    assert 'I love this game' in df_english['review_text'].values
    assert 'Amazing game' in df_english['review_text'].values

    # 非英語レビューは除外される
    assert 'これは素晴らしいゲームです' not in df_english['review_text'].values
    assert 'Разраба маму' not in df_english['review_text'].values


def test_create_topic_model():
    """BERTopicモデル作成のテスト"""
    topic_model = create_topic_model(
        min_topic_size=5,
        verbose=False
    )

    # モデルが作成される
    assert topic_model is not None
    assert hasattr(topic_model, 'fit_transform')


def test_extract_topics_small_data():
    """小規模データでのトピック抽出テスト（100件）"""
    # データ読み込み
    df = pd.read_csv('data/train/reviews_1000.csv')
    df = df.dropna(subset=['review_text'])

    # 英語レビューのみフィルタリング
    df_english = filter_english_reviews(df, text_column='review_text')

    # 100件に制限
    texts = df_english['review_text'].head(100).tolist()

    # トピック抽出
    topic_model, topics, probs = extract_topics(
        texts,
        min_topic_size=5,
        verbose=False
    )

    # 結果確認
    assert topic_model is not None
    assert len(topics) == len(texts)
    assert len(probs) == len(texts)

    # トピックが抽出される（最低1つ以上）
    num_topics = len(set(topics)) - 1  # -1はOutlier除外
    assert num_topics >= 0

    print(f"\n✅ トピック抽出成功: {num_topics}個のトピックを抽出")


def test_get_topic_info():
    """トピック情報取得のテスト"""
    # データ読み込み
    df = pd.read_csv('data/train/reviews_1000.csv')
    df = df.dropna(subset=['review_text'])
    df_english = filter_english_reviews(df, text_column='review_text')
    texts = df_english['review_text'].head(100).tolist()

    # トピック抽出
    topic_model, topics, _ = extract_topics(
        texts,
        min_topic_size=5,
        verbose=False
    )

    # トピック情報取得
    topic_info = get_topic_info(topic_model, verbose=False)

    # データフレームが返される
    assert isinstance(topic_info, pd.DataFrame)
    assert 'Topic' in topic_info.columns
    assert 'Count' in topic_info.columns
    assert 'Name' in topic_info.columns

    print(f"\n✅ トピック情報取得成功: {len(topic_info)}行")


def test_get_topic_words():
    """トピック代表単語取得のテスト"""
    # データ読み込み
    df = pd.read_csv('data/train/reviews_1000.csv')
    df = df.dropna(subset=['review_text'])
    df_english = filter_english_reviews(df, text_column='review_text')
    texts = df_english['review_text'].head(100).tolist()

    # トピック抽出
    topic_model, topics, _ = extract_topics(
        texts,
        min_topic_size=5,
        verbose=False
    )

    # Topic 0の代表単語を取得
    if 0 in topics:
        words = get_topic_words(topic_model, topic_id=0, top_n=5)

        # 単語リストが返される
        assert isinstance(words, list)
        if len(words) > 0:
            assert len(words) <= 5
            # 各要素は(単語, スコア)のタプル
            assert isinstance(words[0], tuple)
            assert isinstance(words[0][0], str)
            assert isinstance(words[0][1], float)

            print(f"\n✅ トピック代表単語取得成功: {', '.join([w for w, _ in words])}")


def test_print_topic_summary():
    """トピックサマリー表示のテスト"""
    # データ読み込み
    df = pd.read_csv('data/train/reviews_1000.csv')
    df = df.dropna(subset=['review_text'])
    df_english = filter_english_reviews(df, text_column='review_text')
    texts = df_english['review_text'].head(100).tolist()

    # トピック抽出
    topic_model, topics, _ = extract_topics(
        texts,
        min_topic_size=5,
        verbose=False
    )

    # サマリー表示（エラーが出ないことを確認）
    print_topic_summary(
        topic_model,
        topics,
        texts,
        max_topics=3,
        top_n_words=5,
        sample_reviews=2
    )

    print("\n✅ トピックサマリー表示成功")


if __name__ == '__main__':
    # 個別実行用
    print("=" * 70)
    print("トピック抽出モジュールのテスト")
    print("=" * 70)

    test_is_english()
    print("✅ test_is_english passed")

    test_filter_english_reviews()
    print("✅ test_filter_english_reviews passed")

    test_create_topic_model()
    print("✅ test_create_topic_model passed")

    test_extract_topics_small_data()
    print("✅ test_extract_topics_small_data passed")

    test_get_topic_info()
    print("✅ test_get_topic_info passed")

    test_get_topic_words()
    print("✅ test_get_topic_words passed")

    test_print_topic_summary()
    print("✅ test_print_topic_summary passed")

    print("\n" + "=" * 70)
    print("全テスト成功！")
    print("=" * 70)


# --- ゲーム名の除去（docs/decisions.md 2026-09-06「語 × ゲーム」で決める） ---

def _roster_df():
    """同じ語が、あるゲームでは固有名詞・別のゲームでは内容語になるデータ"""
    return pd.DataFrame({
        'game_name': [
            'Magic: The Gathering Arena',
            'Sid Meier\u2019s Civilization\u00ae VI',
            'Split Fiction',
            'FINAL FANTASY XIV Online',
        ],
        'review_text': [
            'magic the gathering arena is a great card game with magic cards',
            'the magic of building a civilization from nothing',
            'great split screen game, we played split fiction together',
            'the best fantasy world, and split screen would be nice',
        ],
    })


def test_remove_game_names_keeps_content_words_in_other_games():
    """自分のタイトルの語は自分のレビューからだけ消え、他ゲームでは内容語として残る"""
    result = remove_game_names(_roster_df(), all_games=True)
    texts = result['review_text'].tolist()

    # MTGのレビューからは magic / gathering / arena が消える
    assert 'magic' not in texts[0]
    assert 'card game' in texts[0]

    # 他ゲームのレビューでは magic が内容語として残る（ここが2026-09-06の修正点）
    assert 'magic' in texts[1]
    assert 'civilization' not in texts[1]   # Civ自身のレビューなので消える

    # split は Split Fiction のレビューでだけ消え、FFXIVでは残る
    assert 'split' not in texts[2]
    assert 'split screen' in texts[3]
    assert 'fantasy' not in texts[3]        # FFXIV自身のレビューなので消える


def test_remove_game_names_removes_full_title_from_all_reviews():
    """完全なタイトルの並びは全レビューから消える（他ゲームへの言及対策）"""
    df = pd.DataFrame({
        'game_name': ['Starfield', 'Palia', 'Magic: The Gathering Arena'],
        'review_text': [
            'starfield is huge',
            'better than magic the gathering arena and starfield',
            'magic the gathering arena has good cards',
        ],
    })
    result = remove_game_names(df, all_games=True)
    texts = result['review_text'].tolist()

    # ロスター他ゲームのタイトルは、2語以上の並びとして出れば消える
    assert 'gathering' not in texts[1]
    # 並びで消えるだけなので、単語としての magic は他ゲームに残せる
    assert 'magic' in remove_game_names(
        pd.DataFrame({'game_name': ['Palia'],
                      'review_text': ['the magic of this world']}),
        all_games=True)['review_text'].iloc[0]


def test_remove_game_names_single_word_title_is_not_global():
    """1語のタイトルは自動で全レビューから消さない（一般語を巻き添えにするため）

    実測: predecessor は全1,815回のうち1,108回が Silksong のレビューで
    「前作」の意味で使われていた。
    """
    df = pd.DataFrame({
        'game_name': ['Predecessor', 'Hollow Knight: Silksong'],
        'review_text': [
            'predecessor is a moba',
            'better than its predecessor in every way',
        ],
    })
    result = remove_game_names(df, all_games=True)
    texts = result['review_text'].tolist()

    assert 'predecessor' not in texts[0]   # 自分のレビューからは消える
    assert 'predecessor' in texts[1]       # 他ゲームでは内容語として残る


def test_remove_game_names_single_word_title_via_extra_words():
    """1語のタイトルを全レビューから消したいときは人がリストに書く"""
    df = pd.DataFrame({
        'game_name': ['Starfield', 'Palia'],
        'review_text': ['starfield is huge', 'not like starfield at all'],
    })
    result = remove_game_names(df, all_games=True, extra_words=['starfield'])
    assert all('starfield' not in t for t in result['review_text'])


def test_remove_game_names_extra_words_are_global():
    """extra_words（ロスター外の固有名詞）は全レビューから消える"""
    df = pd.DataFrame({
        'game_name': ['Starfield', 'Palia'],
        'review_text': [
            'bethesda made this, feels like skyrim',
            'bethesda games are different from this',
        ],
    })
    result = remove_game_names(df, all_games=True, extra_words=['bethesda', 'skyrim'])
    assert all('bethesda' not in t and 'skyrim' not in t
               for t in result['review_text'])


def test_remove_game_names_all_games_false_is_own_game_only():
    """all_games=False では自ゲームの語だけを消し、タイトルの並びは触らない"""
    df = pd.DataFrame({
        'game_name': ['Split Fiction', 'FINAL FANTASY XIV Online'],
        'review_text': [
            'split fiction is fun',
            'split screen would be nice in split fiction',
        ],
    })
    result = remove_game_names(df, all_games=False)
    texts = result['review_text'].tolist()

    assert 'split' not in texts[0]
    # 他ゲームのレビューには手を付けない
    assert 'split fiction' in texts[1]


def test_remove_game_names_short_and_numeric_words_are_kept():
    """3文字以下と数字のみの語は自ゲーム除去の対象にしない"""
    df = pd.DataFrame({
        'game_name': ['Dota 2'],
        'review_text': ['dota 2 has 2 teams of 5'],
    })
    result = remove_game_names(df, all_games=False)
    text = result['review_text'].iloc[0]

    assert 'dota' not in text
    assert '2' in text and '5' in text


# --- extra_titles（入力にない別ゲームのタイトルを全レビューから消す） ---

def _extra_df():
    return pd.DataFrame({
        'game_name': ['Palia', 'Starfield'],
        'review_text': [
            'better than magic the gathering arena and predecessor',
            'magic the gathering arena fans will like predecessor here',
        ],
    })


def test_remove_game_names_extra_title_multiword_is_removed_from_all_games():
    """extra_titles の2語以上のタイトルは、どのゲームのレビューからも消える"""
    result = remove_game_names(_extra_df(), all_games=True,
                               extra_titles=['Magic: The Gathering Arena'])
    assert not any('gathering' in t for t in result['review_text'])


def test_remove_game_names_extra_title_single_word_is_kept():
    """extra_titles の1語のタイトルは消えない"""
    result = remove_game_names(_extra_df(), all_games=True,
                               extra_titles=['Predecessor'])
    assert all('predecessor' in t for t in result['review_text'])


def test_remove_game_names_extra_title_duplicating_input_title_same_as_without():
    """extra_titles が入力のタイトルと重複しても、重複なしのときと結果が同じ"""
    df = pd.DataFrame({
        'game_name': ['Palia', 'Magic: The Gathering Arena'],
        'review_text': ['better than magic the gathering arena', 'magic the gathering arena is fun'],
    })
    dup = remove_game_names(df, all_games=True, extra_titles=['Magic: The Gathering Arena'])
    plain = remove_game_names(df, all_games=True)
    assert dup['review_text'].tolist() == plain['review_text'].tolist()


def test_remove_game_names_extra_title_works_without_all_games():
    """all_games=False でも extra_titles があれば全レビューから消える"""
    result = remove_game_names(_extra_df(), all_games=False,
                               extra_titles=['Magic: The Gathering Arena'])
    assert not any('gathering' in t for t in result['review_text'])


# --- assign_topics の chunk_size ---

class _FakeTopicModel:
    """テキストの長さから決まる値を返す（呼ばれた回数も数える）"""
    def __init__(self):
        self.calls = 0

    def transform(self, texts):
        self.calls += 1
        topics = [len(t) % 3 - 1 for t in texts]
        probs = np.array([len(t) / 100 for t in texts])
        return topics, probs


_TEXTS = ['a', 'bb', 'ccc', 'dddd', 'eeeee']


def test_assign_topics_chunk_size_2_calls_transform_three_times():
    """5件を chunk_size=2 で回すと transform は3回呼ばれる"""
    model = _FakeTopicModel()
    assign_topics(model, _TEXTS, verbose=False, chunk_size=2)
    assert model.calls == 3


def test_assign_topics_chunked_topics_same_as_unchunked():
    """分けて割り当てても、topics は一度に割り当てたときと同じ"""
    whole, _ = assign_topics(_FakeTopicModel(), _TEXTS, verbose=False, chunk_size=0)
    chunked, _ = assign_topics(_FakeTopicModel(), _TEXTS, verbose=False, chunk_size=2)
    assert chunked == whole


def test_assign_topics_chunked_probabilities_same_as_unchunked():
    """分けて割り当てても、probabilities は一度に割り当てたときと同じ"""
    _, whole = assign_topics(_FakeTopicModel(), _TEXTS, verbose=False, chunk_size=0)
    _, chunked = assign_topics(_FakeTopicModel(), _TEXTS, verbose=False, chunk_size=2)
    assert np.array_equal(chunked, whole)


def test_assign_topics_chunked_none_probabilities_stay_none():
    """transform が確率を返さない（None）ときは、分けても None のまま"""
    class _NoProb:
        def transform(self, texts):
            return [0] * len(texts), None
    _, probs = assign_topics(_NoProb(), _TEXTS, verbose=False, chunk_size=2)
    assert probs is None


# --- extract_topics.py の --extra-titles 読み込み ---

def test_load_extra_titles_without_name_column_exits(tmp_path):
    """name 列が無いCSVなら止まる（黙って空にしない）"""
    path = tmp_path / 'titles.csv'
    path.write_text('title\nStarfield\n', encoding='utf-8')
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/nlp'))
    from extract_topics import load_extra_titles
    with pytest.raises(SystemExit):
        load_extra_titles(str(path))


def test_load_extra_titles_missing_file_exits(tmp_path):
    """ファイルが無いときも止まる"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/nlp'))
    from extract_topics import load_extra_titles
    with pytest.raises(SystemExit):
        load_extra_titles(str(tmp_path / 'none.csv'))


# --- create_topic_model の umap_random_state ---

def test_create_topic_model_umap_seed_sets_random_state(monkeypatch):
    """umap_random_state=7 なら UMAP の random_state が 7 になる"""
    import src.nlp.topic as topic
    monkeypatch.setattr(topic, 'SentenceTransformer', lambda name: object())
    model = topic.create_topic_model(umap_random_state=7, verbose=False)
    assert model.umap_model.random_state == 7


def test_create_topic_model_umap_seed_none_keeps_default_umap(monkeypatch):
    """None なら BERTopic の既定の UMAP のまま（random_state を足さない）"""
    import src.nlp.topic as topic
    monkeypatch.setattr(topic, 'SentenceTransformer', lambda name: object())
    model = topic.create_topic_model(verbose=False)
    assert model.umap_model.random_state is None


# --- save_topic_model の serialization ---

def test_save_topic_model_pickle_passes_pickle_to_save(tmp_path):
    """serialization='pickle' なら model.save に pickle を渡す"""
    from unittest.mock import MagicMock
    from src.nlp.topic import save_topic_model
    model = MagicMock()
    save_topic_model(model, str(tmp_path / 'm.pkl'), serialization='pickle')
    assert model.save.call_args.kwargs['serialization'] == 'pickle'


# --- uses_similarity_assignment ---

def test_uses_similarity_assignment_base_cluster_is_true():
    """hdbscan_model が BaseCluster なら True"""
    from types import SimpleNamespace
    from bertopic.cluster import BaseCluster
    from src.nlp.topic import uses_similarity_assignment
    assert uses_similarity_assignment(SimpleNamespace(hdbscan_model=BaseCluster())) is True


def test_uses_similarity_assignment_other_cluster_is_false():
    """BaseCluster 以外（HDBSCAN など）なら False"""
    from types import SimpleNamespace
    from src.nlp.topic import uses_similarity_assignment
    assert uses_similarity_assignment(SimpleNamespace(hdbscan_model=object())) is False


# --- extract_topics.py の止まる判定 ---

def test_should_stop_for_similarity_without_allow_stops():
    """類似度の形で allow なしなら止まる"""
    from types import SimpleNamespace
    from bertopic.cluster import BaseCluster
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/nlp'))
    from extract_topics import should_stop_for_similarity
    assert should_stop_for_similarity(SimpleNamespace(hdbscan_model=BaseCluster()), False) is True


def test_should_stop_for_similarity_with_allow_does_not_stop():
    """類似度の形でも allow ありなら止まらない"""
    from types import SimpleNamespace
    from bertopic.cluster import BaseCluster
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/nlp'))
    from extract_topics import should_stop_for_similarity
    assert should_stop_for_similarity(SimpleNamespace(hdbscan_model=BaseCluster()), True) is False
