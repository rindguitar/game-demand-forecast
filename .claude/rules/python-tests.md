---
paths:
  - "tests/**/*.py"
  - "src/**/test_*.py"
---

# テストコード規約（テストファイル編集時のみロードされる）

- pytest を使用。unittest スタイルは書かない
- テスト名は `test_<対象>_<条件>_<期待結果>` 形式（例: `test_sentiment_empty_input_raises`）
- 外部API（Steam API等）は必ずモックする。実通信するテストを書かない
- 乱数を使う処理はシード固定で再現性を担保。効果量を主張する実験は複数シードで検証する
- 1テスト1アサーションを原則とし、共通処理は fixture で切り出す

<!--
  paths にマッチするファイルを Claude が触るときだけコンテキストにロードされる。
  常時ロードの CLAUDE.md を太らせずに済む。
  同様に data-pipeline.md, timeseries.md などトピック別に増やしていく。
-->
