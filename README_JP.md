# OpenVoiceChanger

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" alt="Python 3.13" />
  <img src="https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.14" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white" alt="React 19" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License" />
</p>

<p align="center">
  ブラウザで動作するローカルのリアルタイム／オフライン RVC ボイスチェンジャースタジオ。<br/>
  コマンド 1 つで起動、Python の手動セットアップ不要、品質優先のストリーミング。
</p>

<p align="center">
  <a href="README.md">English</a> •
  <a href="README_KR.md">한국어</a>
</p>

---

## クイックスタート

**必要なもの:** Windows、Linux、macOS のいずれか。それ以外は不要です。起動スクリプトが
[uv](https://docs.astral.sh/uv/) をインストールし、uv が Python とすべての依存関係を自動で取得します。
NVIDIA GPU があれば自動で使用され、Apple Silicon では MPS を使用します。

```bash
git clone https://github.com/sioaeko/OpenVoiceChanger.git
cd OpenVoiceChanger
./start.sh          # Windows: start.bat をダブルクリック
```

スタジオは <http://127.0.0.1:8000> で開きます。初回起動時には共通モデルアセット
（HuBERT 約 190 MB、RMVPE 約 180 MB）をダウンロードし、SHA-256 で検証します。
リリースアーカイブには Web UI がビルド済みで同梱されています。git から取得した場合は、
[Node.js](https://nodejs.org/) 20 以上があれば自動でビルドされます。

**Models** タブで RVC の `.pth` ボイス（あれば `.index` も）をドロップし、**Activate** を押して、
**Studio** でボイスチェンジャーを開始します。

```bash
./start.sh --port 8080 --no-browser     # CLI オプションはそのまま渡されます
OVC_TORCH=cpu ./start.sh                # CPU 版 PyTorch を強制（Windows では set OVC_TORCH=cpu）
./start.sh --device cpu                 # GPU 版のまま CPU で推論
```

## 主な機能

- RVC v1/v2（ピッチあり／なし）のリアルタイム変換。バイナリ WebSocket と AudioWorklet でストリーミング
- 品質優先のストリーミング: 各チャンクを **過去 1 秒・未来 150 ms のコンテキスト**付きで再レンダリングし、位相を揃えたクロスフェード（SOLA）で接続
- 設定の 3 つのプロファイルと推定遅延: **Quality**（約 700 ms）、**Balanced**（約 400 ms）、**Responsive**（約 250 ms）
- GPU 上の厳密な近傍検索による **index 検索**。従来の近似検索より声質が近くなります
- ピッチ・フォルマントシフト、RVC 詳細設定、**13 種類のピッチ推定**（PM、Harvest、DIO、Crepe 系、RMVPE、FCPE、オフライン用ハイブリッド）。GPU では RMVPE を自動選択
- **13 種類の DSP エフェクト**（ノイズゲート、ロボット、ウィスパー、電話、ディストーション、ビットクラッシュ、コーラス、エコー、リバーブ、EQ、コンプレッサー、ゲイン、リミッター）
- **16 個の内蔵プリセット**とユーザープリセット、無音時に推論を止める **Silence Saver**
- スペクトラム、メーター、遅延、サーバー処理時間の表示。出力録音（<kbd>R</kbd>）と A/B バイパス（<kbd>B</kbd>）
- オフライン変換: ファイル全体を、RVC 本来のパディングと分割で最高品質に変換

## スクリーンショット

![Studio](docs/images/main-ui.png)
![Presets & effects](docs/images/effects-rack.png)
![Models](docs/images/models.png)
![Converter](docs/images/converter.png)
![Settings](docs/images/settings-modal.png)

## リアルタイムエンジンの仕組み

1. **ローリングウィンドウの再レンダリング:** 直近 1 秒の入力と新しいチャンクを毎回推論し、HuBERT・ピッチ推定・合成器に実際の過去コンテキストを与えます。
2. **先読みの保留:** 各レンダリングの最新 150 ms はまだ出力しません。出力されるすべてのサンプルは、その後ろの実音声とともに推論されています。モデル端の歪みが出力に届かない代わりに、遅延が固定で 150 ms 増えます。
3. **SOLA で接続:** 前回保留した末尾を新しいレンダリングと ±6 ms の範囲で照合し、最も相関の高い位置でクロスフェードします。クリックや位相のうねりが生じません。
4. **soxr による高品質リサンプリング:** ウィンドウ端が回り込む FFT リサンプリングは使いません。
5. **シフト不変の音量エンベロープ:** RMS ミックスをスライディングウィンドウで計算し、重なり合うレンダリング間で音量が一致します。

推論コードは MIT ライセンスの RVC 合成器・RMVPE と fairseq の HuBERT を移植した自己完結型の実装で、
fairseq・faiss・librosa に依存しません。出力は元の実装と照合済みです。

## 設定

すべての設定は環境変数、またはプロジェクトフォルダの `.env` で変更できます。主なもの:

| 変数 | 既定値 | 意味 |
|---|---|---|
| `OVC_HOST` / `OVC_PORT` | `127.0.0.1` / `8000` | 待ち受けアドレス。`0.0.0.0` で LAN に公開 |
| `OVC_DEVICE` | `auto` | `auto`、`cuda`、`mps`、`xpu`、`cpu` |
| `OVC_MODELS_DIR` | `models` | ボイスと `.index` の置き場所 |
| `OVC_CHUNK_SIZE` | `8192` | 既定のチャンクサイズ（48 kHz のサンプル数） |
| `OVC_STREAM_LOOKBACK_SECONDS` | `1.0` | 各チャンクで再推論する過去コンテキスト |
| `OVC_STREAM_LOOKAHEAD_SECONDS` | `0.15` | 未来コンテキスト（この分だけ遅延が増加） |
| `OVC_INDEX_EXACT_SEARCH` | `true` | `false` で従来 RVC の近似検索を再現 |
| `OVC_RVC_ALLOW_UNSAFE_CHECKPOINTS` | `false` | 下記「セキュリティ」を参照 |

全項目は [backend/config.py](backend/config.py) を参照してください。

## セキュリティ

- 既定ではループバックのみで待ち受けます。クロスオリジンのアクセスは `OVC_CORS_ORIGINS` に限られ、WebSocket は `Origin` ヘッダーを検証します。
- `.pth` は Python の pickle で、読み込むとコードが実行される可能性があります。PyTorch の `weights_only` で読み込み、それ以上が必要なチェックポイントは `OVC_RVC_ALLOW_UNSAFE_CHECKPOINTS=true` を設定しない限り拒否します。

## 開発

```bash
make install        # uv sync（CPU 版 torch。CUDA は TORCH=cuda）+ npm ci
make dev-backend    # :8000 で API を自動リロード起動
make dev-frontend   # :5173 で Vite 開発サーバー
make lint test      # ruff + pyright + eslint、pytest + vitest
```

## ライセンス

MIT
