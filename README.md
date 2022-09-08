# Stable Diffusion

## 機能

- ローカル環境、Google Colab 環境それぞれの Notebook を提供
  - txt2img, img2img に対応
  - prompt_correction パッチ ([参考](https://zenn.dev/td2sk/articles/eb772103a3a8ff)) を実装済み
- Colab 無料プランでの WaifuDiffusion full-ema モデル読み込み対応
  - 動作確認済みモデル
    - Stable Diffusion (sd-v1-4.ckpt)
    - Waifu Diffusion (wd-v1-2-full-ema.ckpt)

## 実行

### Google Colab で実行

#### txt2img_colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/td2sk/stable-diffusion/blob/main/txt2img_colab.ipynb)

#### img2img_colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/td2sk/stable-diffusion/blob/main/img2img_colab.ipynb)

### ローカルで実行

#### txt2img

[txt2img.ipynb](./txt2img.ipynb)

#### img2img

[img2img.ipynb](./img2img.ipynb)

## ライセンス

- Stable Diffusion からフォークしたコード、Stable Diffusion から派生したコードのライセンスは Stable Diffusion のライセンスに準ずる
- Notebook のライセンスは CC0 とする

## 更新履歴

- 2022/09/07
  - 公開
- 2022/09/08
  - Colab 無料枠で Waifu Diffusion (wd-v1-2-full-ema.cpkt) が動作するよう修正
