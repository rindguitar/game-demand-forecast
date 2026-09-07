---
paths:
  - "**/README.md"
  - "docs/**/*.md"
---

# GitHub Wiki 構成規約

<!--
  Wikiは本体リポジトリとは別のGitリポジトリなので、paths による自動ロードが
  効かない場合がある。CLAUDE.md の「ドキュメント規約」から明示的に参照されるため、
  Wikiを触る指示を受けたらこのファイルを読むこと。
-->

## 何をWikiに書くか

| 置き場所 | 内容 |
|---|---|
| GitHub Wiki | 技術概念の解説、調査・検証レポート、分析結果 |
| `README.md`（各ディレクトリ） | そのディレクトリの構成・処理の流れ・使い方 |
| `docs/decisions.md` | 設計判断の記録（なぜそれを選んだか） |
| `docs/STATUS.md` | 現在地・次の一手 |

**1ページ1概念**。1つのページに複数の概念を詰めない。

## ページの命名と表記

- ページ名（ファイル名）は**英語のケバブケース**: `Data-Leakage.md`, `Cross-Validation.md`
- Homeからのリンクは**日本語の表示名 + 1行説明**を付ける

```markdown
- **[データリーケージ（Data Leakage）](Data-Leakage)** - 評価の公平性が壊れる現象・3パターン・防ぎ方
```

- 略称は単独で使わず、日本語と併記する（例: `ドメイン適応事前学習（DAPT）`）

## Home.md の構成

Homeはハブとして機能させる。構成は以下の順:

1. **冒頭**: プロジェクトの1行説明 + ドキュメントマップへのリンク
2. **技術ドキュメント**: ジャンルごとに `<details>` で折りたたむ
3. **プロジェクト情報**: README・リポジトリ構成など外部への導線
4. **更新履歴**: `<details>` で折りたたむ。追加・改訂を1行ずつ

```markdown
<details>
<summary><b>ジャンル名</b> — このジャンルが何を扱うかの一言</summary>

- **[表示名](Page-Name)** - 1行説明
- **[表示名](Page-Name)** - 1行説明

</details>
```

### ジャンル分けの原則

**「分野を問わず通じる汎用知識」と「このプロジェクト固有の調査・判断」を必ず分ける。**
前者は他プロジェクトでも再利用でき、後者は文脈が変われば無効になるため、混ぜると
どれが持ち出せる知識なのか分からなくなる。

ジャンル例（プロジェクトに合わせて変える）:
- 分野の基礎知識（学習・評価・データ設計・実験の進め方）
- ドメイン別の技術（NLP、時系列、可視化など）
- ライブラリ・ツールの仕組み
- 環境・ハードウェア
- プロジェクト設計・検証レポート ← 固有のものはここに隔離

## ドキュメントマップ

全ページのリンク関係を1枚の図にしたページ（`Documentation-Map`）を置く。
用途は「どこから読むか」と「**どのページが孤立しているか**」の把握。

- IMPORTANT: ページを追加したりリンクを張り替えたら、ドキュメントマップも作り直す。
  元データが変わると図が実態とずれ、孤立ページ検出という本来の用途が壊れる
- 図の作り方は `.claude/rules/mermaid.md` に従う（特に「まとめる → 分ける」の2階層化）

### 作り直しの手順

```bash
# 1. Wikiをcloneする（<owner>/<repo> は対象リポジトリに置き換える）
git clone https://github.com/<owner>/<repo>.wiki.git /tmp/wiki
```

```python
# 2. ページ間のリンクを抽出する
#    各ページ本文の ](ページ名) を走査し、Wiki内のページ名と一致するものだけ拾う。
#    向きは落とし、往復しているものは1本に畳む。
import os, re
d = '/tmp/wiki'
skip = {'Home', 'Documentation-Map'}
pages = {f[:-3]: open(os.path.join(d, f), encoding='utf-8').read()
         for f in os.listdir(d) if f.endswith('.md')}
edges = {tuple(sorted((src, t.split('#')[0].strip())))
         for src, txt in pages.items() if src not in skip
         for t in re.findall(r'\]\(([^)]+)\)', txt)
         if t.split('#')[0].strip() in pages
         and t.split('#')[0].strip() not in skip | {src}}
print(len(pages), 'ページ /', len(edges), '本')
```

3. **密に繋がるまとまりを見つける**: 貪欲モジュラリティ最大化（隣り合うまとまりを、
   モジュラリティが上がる限りくっつけていく）でブロックを検出する。
   **分類を人が決めない**ことが大事で、そうすることで「Homeの分類と実際のリンクがずれている」
   といった発見が出る
4. **図を組む**: 全体地図 → ブロックごとの内訳、の2階層にする。ブロックの名前だけは
   中身を見て人が付ける
5. **画像に描き出して、全枚数を目視する**（省略しない。1枚でも貫通していれば誤読される）
6. pushして、Homeの更新履歴に1行足す

```bash
cd /tmp/wiki && git add -A && git commit -m "docs: ..." && git push
```

<!--
  構成の実例:
  https://github.com/rindguitar/game-demand-forecast/wiki
-->
