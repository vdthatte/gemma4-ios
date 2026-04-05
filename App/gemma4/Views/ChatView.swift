//
//  ChatView.swift
//  gemma4
//

import SwiftUI

struct TypingIndicator: View {
    @State private var phase = 0.0

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { i in
                Circle()
                    .fill(Color.secondary.opacity(0.5))
                    .frame(width: 7, height: 7)
                    .offset(y: sin(phase + Double(i) * 0.8) * 3)
            }
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 0.6).repeatForever(autoreverses: false)) {
                phase = .pi * 2
            }
        }
    }
}

struct ChatView: View {
    @Bindable var viewModel: ChatViewModel
    @FocusState private var isInputFocused: Bool

    var body: some View {
        if let conversation = viewModel.currentConversation {
            VStack(spacing: 0) {
                // Messages
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 16) {
                            ForEach(conversation.sortedMessages.filter { $0.role != .system }) { message in
                                MessageBubbleView(message: message)
                                    .id(message.id)
                                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                            }

                            if viewModel.isGenerating,
                               let last = conversation.sortedMessages.last,
                               last.content.isEmpty {
                                HStack {
                                    Circle()
                                        .fill(
                                            LinearGradient(
                                                colors: [.blue.opacity(0.6), .purple.opacity(0.6)],
                                                startPoint: .topLeading,
                                                endPoint: .bottomTrailing
                                            )
                                        )
                                        .frame(width: 28, height: 28)
                                        .overlay {
                                            Image(systemName: "sparkles")
                                                .font(.system(size: 13, weight: .semibold))
                                                .foregroundStyle(.white)
                                        }
                                    TypingIndicator()
                                        .padding(.horizontal, 14)
                                        .padding(.vertical, 12)
                                        .background(Color(.systemGray6))
                                        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                                    Spacer()
                                }
                                .padding(.horizontal, 4)
                            }
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                    }
                    .scrollDismissesKeyboard(.interactively)
                    .onTapGesture { isInputFocused = false }
                    .onChange(of: conversation.sortedMessages.last?.content) {
                        if let lastId = conversation.sortedMessages.last?.id {
                            withAnimation(.easeOut(duration: 0.15)) {
                                proxy.scrollTo(lastId, anchor: .bottom)
                            }
                        }
                    }
                }

                // Status bar
                if viewModel.isGenerating || viewModel.errorMessage != nil {
                    HStack(spacing: 6) {
                        if viewModel.isGenerating {
                            Image(systemName: "bolt.fill")
                                .font(.caption2)
                                .foregroundStyle(.blue)
                            Text("\(String(format: "%.1f", viewModel.tokensPerSecond)) tok/s")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        if let error = viewModel.errorMessage {
                            Image(systemName: "exclamationmark.circle.fill")
                                .font(.caption2)
                                .foregroundStyle(.red)
                            Text(error)
                                .font(.caption2)
                                .foregroundStyle(.red)
                                .lineLimit(1)
                        }
                        Spacer()
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 4)
                }

                Divider()

                // Input
                inputBar
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
            }
            // nav title handled by ContentView toolbar
        } else {
            ContentUnavailableView(
                "No Chat Selected",
                systemImage: "bubble.left.and.bubble.right",
                description: Text("Create a new chat to get started")
            )
        }
    }

    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 0) {
            TextField("Message...", text: $viewModel.prompt, axis: .vertical)
                .focused($isInputFocused)
                .lineLimit(1...6)
                .padding(.leading, 16)
                .padding(.trailing, 6)
                .padding(.vertical, 10)
                .onSubmit { sendIfPossible() }

            Button(action: sendOrStop) {
                Image(systemName: viewModel.isGenerating
                    ? "stop.circle.fill"
                    : "arrow.up.circle.fill"
                )
                .font(.system(size: 28))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(canSend || viewModel.isGenerating ? .blue : Color(.systemGray3))
            }
            .disabled(!canSend && !viewModel.isGenerating)
            .animation(.easeInOut(duration: 0.15), value: canSend)
            .padding(.trailing, 6)
            .padding(.bottom, 4)
        }
        .background(Color(.systemGray6))
        .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .strokeBorder(Color(.systemGray4), lineWidth: 0.5)
        )
    }

    private var canSend: Bool {
        !viewModel.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !viewModel.isGenerating
    }

    private func sendIfPossible() {
        guard canSend else { return }
        Task { await viewModel.sendMessage() }
    }

    private func sendOrStop() {
        if viewModel.isGenerating {
            viewModel.stopGenerating()
        } else {
            isInputFocused = false
            Task { await viewModel.sendMessage() }
        }
    }
}
