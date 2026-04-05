//
//  MLXService.swift
//  gemma4
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
@preconcurrency import Hub

extension HubApi {
    #if os(macOS)
    static let `default` = HubApi(
        downloadBase: URL.downloadsDirectory.appending(path: "huggingface")
    )
    #else
    static let `default` = HubApi(
        downloadBase: URL.cachesDirectory.appending(path: "huggingface")
    )
    #endif
}

@Observable
class MLXService {

    private var modelContainers: [GemmaModel: ModelContainer] = [:]
    private var registeredGemma4 = false

    private static let lastModelKey = "lastUsedModel"

    func isModelLoaded(_ model: GemmaModel) -> Bool {
        modelContainers[model] != nil
    }

    var lastUsedModel: GemmaModel? {
        get {
            guard let raw = UserDefaults.standard.string(forKey: Self.lastModelKey) else { return nil }
            return GemmaModel(rawValue: raw)
        }
        set {
            UserDefaults.standard.set(newValue?.rawValue, forKey: Self.lastModelKey)
        }
    }

    private func registerGemma4IfNeeded() async {
        guard !registeredGemma4 else { return }
        registeredGemma4 = true

        await LLMTypeRegistry.shared.registerModelType("gemma4") { data in
            let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: data)
            return Gemma4TextModel(config)
        }
        await LLMTypeRegistry.shared.registerModelType("gemma4_text") { data in
            let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: data)
            return Gemma4TextModel(config)
        }
    }

    /// Load model — uses cached files on disk if available (no re-download).
    func loadModel(_ model: GemmaModel, onProgress: (@Sendable (Double) -> Void)? = nil) async throws {
        if modelContainers[model] != nil { return }

        Memory.cacheLimit = 20 * 1024 * 1024
        await registerGemma4IfNeeded()

        let configuration = ModelConfiguration(id: model.modelId)
        let container = try await LLMModelFactory.shared.loadContainer(
            hub: .default,
            configuration: configuration
        ) { progress in
            onProgress?(progress.fractionCompleted)
        }

        modelContainers[model] = container
        lastUsedModel = model
    }

    func generate(messages: [ChatMessage], model: GemmaModel) async throws -> AsyncStream<Generation> {
        guard let modelContainer = modelContainers[model] else {
            throw MLXServiceError.modelNotLoaded
        }

        let chat = messages.map { message in
            let role: Chat.Message.Role = switch message.role {
            case .assistant: .assistant
            case .user: .user
            case .system: .system
            }
            return Chat.Message(role: role, content: message.content)
        }

        let userInput = UserInput(chat: chat)

        return try await modelContainer.perform { (context: ModelContext) in
            let lmInput = try await context.processor.prepare(input: userInput)
            let parameters = GenerateParameters(temperature: 0.7)
            return try MLXLMCommon.generate(
                input: lmInput, parameters: parameters, context: context
            )
        }
    }

    enum MLXServiceError: LocalizedError {
        case modelNotLoaded

        var errorDescription: String? {
            switch self {
            case .modelNotLoaded:
                "Model is not loaded. Please download the model first."
            }
        }
    }
}
