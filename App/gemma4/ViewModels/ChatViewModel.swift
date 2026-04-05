//
//  ChatViewModel.swift
//  gemma4
//

import Foundation
import MLXLMCommon
import SwiftData

typealias SDModelContext = SwiftData.ModelContext

enum ModelState: Equatable {
    case notLoaded
    case loading
    case downloading(progress: Double)
    case ready
    case error(String)
}

@Observable
@MainActor
class ChatViewModel {
    private let mlxService: MLXService
    private let modelContext: SDModelContext

    // Model
    var selectedModel: GemmaModel = .e2b
    var modelStates: [GemmaModel: ModelState] = [
        .e2b: .notLoaded,
        .e4b: .notLoaded,
    ]

    var currentModelState: ModelState {
        modelStates[selectedModel] ?? .notLoaded
    }

    // Conversations
    var conversations: [Conversation] = []
    var currentConversation: Conversation?

    // Generation
    var prompt: String = ""
    var isGenerating = false
    var errorMessage: String?
    var tokensPerSecond: Double = 0

    private var generateTask: Task<Void, any Error>?

    init(mlxService: MLXService, modelContext: SDModelContext) {
        self.mlxService = mlxService
        self.modelContext = modelContext

        if let last = mlxService.lastUsedModel {
            selectedModel = last
        }

        loadConversations()

        if conversations.isEmpty {
            createNewChat()
        } else {
            currentConversation = conversations.first
        }
    }

    // MARK: - Persistence

    private func loadConversations() {
        let descriptor = FetchDescriptor<Conversation>(
            sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
        )
        conversations = (try? modelContext.fetch(descriptor)) ?? []
    }

    private func save() {
        try? modelContext.save()
    }

    // MARK: - Auto-load cached model on launch

    func autoLoadIfCached() async {
        guard currentModelState == .notLoaded else { return }
        modelStates[selectedModel] = .loading

        do {
            try await mlxService.loadModel(selectedModel)
            modelStates[selectedModel] = .ready
        } catch {
            modelStates[selectedModel] = .notLoaded
        }
    }

    // MARK: - Model

    func downloadModel() async {
        modelStates[selectedModel] = .downloading(progress: 0)

        do {
            try await mlxService.loadModel(selectedModel) { [weak self] progress in
                Task { @MainActor in
                    self?.modelStates[self?.selectedModel ?? .e2b] = .downloading(progress: progress)
                }
            }
            modelStates[selectedModel] = .ready
        } catch {
            modelStates[selectedModel] = .error(error.localizedDescription)
        }
    }

    // MARK: - Conversations

    @discardableResult
    func createNewChat() -> Conversation {
        stopGenerating()
        let conversation = Conversation()

        let systemMsg = ChatMessage.system("You are a helpful assistant.", sortOrder: 0)
        conversation.messages.append(systemMsg)

        modelContext.insert(conversation)
        save()

        conversations.insert(conversation, at: 0)
        currentConversation = conversation
        prompt = ""
        errorMessage = nil
        tokensPerSecond = 0
        return conversation
    }

    func selectConversation(_ conversation: Conversation) {
        stopGenerating()
        currentConversation = conversation
        prompt = ""
        errorMessage = nil
        tokensPerSecond = 0
    }

    func deleteConversation(_ conversation: Conversation) {
        modelContext.delete(conversation)
        save()

        conversations.removeAll { $0.id == conversation.id }
        if currentConversation?.id == conversation.id {
            currentConversation = conversations.first
            if currentConversation == nil {
                createNewChat()
            }
        }
    }

    // MARK: - Generation

    func sendMessage() async {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, let conversation = currentConversation else { return }

        if let existing = generateTask {
            existing.cancel()
            generateTask = nil
        }

        isGenerating = true
        errorMessage = nil

        let nextOrder = conversation.messages.count

        let userMsg = ChatMessage.user(text, sortOrder: nextOrder)
        conversation.messages.append(userMsg)

        let assistantMsg = ChatMessage.assistant("", sortOrder: nextOrder + 1)
        conversation.messages.append(assistantMsg)

        save()
        prompt = ""

        generateTask = Task {
            for await generation in try await mlxService.generate(
                messages: conversation.sortedMessages, model: selectedModel
            ) {
                switch generation {
                case .chunk(let chunk):
                    assistantMsg.content += chunk
                case .info(let info):
                    tokensPerSecond = info.tokensPerSecond
                case .toolCall:
                    break
                }
            }
        }

        do {
            try await withTaskCancellationHandler {
                try await generateTask?.value
            } onCancel: {
                Task { @MainActor in
                    generateTask?.cancel()
                }
            }
        } catch {
            errorMessage = error.localizedDescription
        }

        save()
        isGenerating = false
        generateTask = nil
    }

    func stopGenerating() {
        generateTask?.cancel()
        generateTask = nil
        isGenerating = false
        save()
    }
}
