# Gemma 4 iOS

An iOS chat app that runs Google's Gemma 4 language models entirely on-device using Apple's [MLX](https://github.com/ml-explore/mlx-swift) framework. No server, no API keys — just local inference on your iPhone or iPad.

## Features

- **On-device inference** — Models run locally via MLX, keeping your conversations private
- **Two model options** — Gemma 4 E2B (2B params, fastest) and E4B (4B params, smarter)
- **Conversation history** — Persisted locally with SwiftData
- **Streaming responses** — Token-by-token output as the model generates

## Requirements

- Xcode 16+
- iOS 18.0+
- A physical device with an Apple Silicon chip (iPhone 15 Pro or later recommended)
- ~2–4 GB of free storage for model downloads

> **Note:** The Simulator does not support MLX acceleration. Use a physical device for usable performance.

## Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/vdthatte/gemma4-ios.git
   ```
2. Open `App/gemma4.xcodeproj` in Xcode.
3. Select your physical device as the run destination.
4. Build and run. On first launch, the app will prompt you to download a model from HuggingFace (~1–2 GB).

## Architecture

```
App/gemma4/
├── Models/          # Data models (ChatMessage, Conversation, GemmaModel)
├── Services/        # MLXService — model loading & text generation
├── ViewModels/      # ChatViewModel — orchestrates UI state & inference
└── Views/           # SwiftUI views (ChatView, MessageBubbleView)
```

MVVM with SwiftUI and SwiftData. `MLXService` wraps [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-examples) for model management and generation.

## Models

Models are 4-bit quantized variants from the [mlx-community](https://huggingface.co/mlx-community) on HuggingFace:

| Model | Params | HuggingFace ID |
|-------|--------|----------------|
| Gemma 4 E2B | 2B | `mlx-community/gemma-4-E2B-it-4bit` |
| Gemma 4 E4B | 4B | `mlx-community/gemma-4-e4b-it-4bit` |

## License

[MIT](LICENSE)
