"""
データセット分割モジュール

Train/Validation/Testに分割する機能を提供。
- Train: 学習用（70%）
- Validation: ハイパーパラメータ調整・過学習検出用（15%）
- Test: 最終評価用（15%）
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def split_train_val_test(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    データセットをTrain/Val/Testに分割

    Args:
        df: 分割対象のDataFrame（'label'列必須）
        train_ratio: Train比率（デフォルト70%）
        val_ratio: Validation比率（デフォルト15%）
        test_ratio: Test比率（デフォルト15%）
        random_state: 乱数シード（再現性確保）

    Returns:
        (train_df, val_df, test_df)のタプル

    Raises:
        ValueError: 比率の合計が1.0でない、または'label'列がない場合

    Example:
        >>> df = pd.read_csv('data/train/reviews_1000.csv')
        >>> train_df, val_df, test_df = split_train_val_test(df)
        >>> print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        Train: 700, Val: 150, Test: 150
    """
    # バリデーション
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"比率の合計が1.0ではありません: "
            f"{train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}"
        )

    if 'label' not in df.columns:
        raise ValueError("DataFrameに'label'列が必要です")

    # Train+Val / Test に分割（stratifyでラベルバランス維持）
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df['label'],
        random_state=random_state
    )

    # Train / Val に分割
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=train_val_df['label'],
        random_state=random_state
    )

    # インデックスリセット
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def save_split_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = 'data/train'
) -> None:
    """
    分割したdatasetをCSVファイルに保存

    Args:
        train_df: Train DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: 保存先ディレクトリ

    Example:
        >>> save_split_datasets(train_df, val_df, test_df)
        ✅ Train: data/train/train_700.csv (700件)
        ✅ Val: data/train/val_150.csv (150件)
        ✅ Test: data/train/test_150.csv (150件)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # ファイルパス生成
    train_path = os.path.join(output_dir, f'train_{len(train_df)}.csv')
    val_path = os.path.join(output_dir, f'val_{len(val_df)}.csv')
    test_path = os.path.join(output_dir, f'test_{len(test_df)}.csv')

    # 保存
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Train: {train_path} ({len(train_df)}件)")
    print(f"✅ Val: {val_path} ({len(val_df)}件)")
    print(f"✅ Test: {test_path} ({len(test_df)}件)")


def print_split_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """
    分割後の統計情報を表示

    Args:
        train_df: Train DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame

    Example:
        >>> print_split_statistics(train_df, val_df, test_df)
        ============================================================
        データセット分割統計
        ============================================================
        ...
    """
    print("=" * 60)
    print("データセット分割統計")
    print("=" * 60)

    datasets = [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df)
    ]

    for name, df in datasets:
        n_positive = (df['label'] == 1).sum()
        n_negative = (df['label'] == 0).sum()
        total = len(df)

        print(f"\n【{name}】")
        print(f"  総件数: {total}")
        print(f"  Positive: {n_positive} ({n_positive/total:.1%})")
        print(f"  Negative: {n_negative} ({n_negative/total:.1%})")

    print("\n" + "=" * 60)
