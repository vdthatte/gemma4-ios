//
//  ContentView.swift
//  gemma4
//

import SwiftUI

struct ContentView: View {
    @Bindable var viewModel: ChatViewModel
    @State private var showChatsPopover = false
    @State private var showModelPopover = false

    var body: some View {
        NavigationStack {
            Group {
                switch viewModel.currentModelState {
                case .ready:
                    ChatView(viewModel: viewModel)
                case .loading:
                    loadingView
                case .downloading(let progress):
                    downloadProgressView(progress: progress)
                case .notLoaded:
                    downloadPromptView
                case .error(let message):
                    errorView(message: message)
                }
            }
            .animation(.easeInOut(duration: 0.3), value: viewModel.currentModelState)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        showChatsPopover = true
                    } label: {
                        Image(systemName: "bubble.left.and.bubble.right")
                            .font(.body.weight(.medium))
                    }
                    .popover(isPresented: $showChatsPopover) {
                        chatsPopoverContent
                    }
                }
                ToolbarItem(placement: .principal) {
                    Text(viewModel.currentConversation?.displayTitle ?? "New Chat")
                        .font(.headline)
                        .lineLimit(1)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showModelPopover = true
                    } label: {
                        modelBadge
                    }
                    .popover(isPresented: $showModelPopover) {
                        modelPopoverContent
                    }
                }
            }
        }
        .task {
            await viewModel.autoLoadIfCached()
        }
    }

    // MARK: - Chats Popover

    private var chatsPopoverContent: some View {
        NavigationStack {
            List {
                Button {
                    viewModel.createNewChat()
                    showChatsPopover = false
                } label: {
                    Label("New Chat", systemImage: "plus.circle.fill")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.blue)
                }

                ForEach(viewModel.conversations) { conversation in
                    Button {
                        viewModel.selectConversation(conversation)
                        showChatsPopover = false
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 3) {
                                Text(conversation.displayTitle)
                                    .font(.subheadline.weight(.medium))
                                    .foregroundStyle(.primary)
                                    .lineLimit(1)
                                Text(conversation.createdAt, style: .relative)
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                            }
                            Spacer()
                            if conversation.id == viewModel.currentConversation?.id {
                                Image(systemName: "checkmark")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.blue)
                            }
                        }
                    }
                }
                .onDelete { offsets in
                    for offset in offsets {
                        viewModel.deleteConversation(viewModel.conversations[offset])
                    }
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Chats")
            .navigationBarTitleDisplayMode(.inline)
        }
        .frame(minWidth: 300, minHeight: 350)
        .presentationCompactAdaptation(.popover)
    }

    // MARK: - Model Popover

    private var modelBadge: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(viewModel.currentModelState == .ready ? .green : .orange)
                .frame(width: 7, height: 7)
            Text(viewModel.selectedModel.rawValue.uppercased())
                .font(.caption.weight(.bold).monospaced())
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.ultraThinMaterial, in: Capsule())
    }

    private var modelPopoverContent: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(GemmaModel.allCases) { model in
                let state = viewModel.modelStates[model] ?? .notLoaded
                let isSelected = viewModel.selectedModel == model

                Button {
                    viewModel.selectedModel = model
                    if state == .notLoaded {
                        Task { await viewModel.downloadModel() }
                    } else if case .ready = state {
                        showModelPopover = false
                    }
                } label: {
                    HStack(spacing: 12) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(model.displayName)
                                .font(.subheadline.weight(.medium))
                                .foregroundStyle(.primary)
                            Text(model.description)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        modelStateIcon(state, isSelected: isSelected)
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(isSelected ? Color.blue.opacity(0.08) : .clear)
                }

                if model != GemmaModel.allCases.last {
                    Divider().padding(.leading, 16)
                }
            }
        }
        .frame(width: 260)
        .presentationCompactAdaptation(.popover)
    }

    @ViewBuilder
    private func modelStateIcon(_ state: ModelState, isSelected: Bool) -> some View {
        switch state {
        case .ready:
            if isSelected {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.blue)
            } else {
                Image(systemName: "checkmark.circle")
                    .foregroundStyle(.green)
            }
        case .downloading(let progress):
            ZStack {
                Circle()
                    .stroke(Color.blue.opacity(0.2), lineWidth: 2)
                    .frame(width: 20, height: 20)
                Circle()
                    .trim(from: 0, to: progress)
                    .stroke(Color.blue, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                    .frame(width: 20, height: 20)
                    .rotationEffect(.degrees(-90))
            }
        case .loading:
            ProgressView()
                .controlSize(.small)
        case .notLoaded:
            Image(systemName: "arrow.down.circle")
                .foregroundStyle(.secondary)
        case .error:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
                .font(.caption)
        }
    }

    // MARK: - Download / Loading States

    private var loadingView: some View {
        VStack(spacing: 16) {
            ProgressView()
                .controlSize(.large)
            Text("Loading model...")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("Using cached files")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
    }

    private var downloadPromptView: some View {
        VStack(spacing: 24) {
            Spacer()
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [.blue.opacity(0.15), .purple.opacity(0.15)],
                            startPoint: .topLeading, endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 100, height: 100)
                Image(systemName: "arrow.down.circle.fill")
                    .font(.system(size: 44))
                    .foregroundStyle(
                        LinearGradient(colors: [.blue, .purple], startPoint: .topLeading, endPoint: .bottomTrailing)
                    )
            }
            VStack(spacing: 8) {
                Text(viewModel.selectedModel.displayName)
                    .font(.title2.weight(.semibold))
                Text("Download the model to start chatting.\nRuns entirely on your device.")
                    .font(.subheadline)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
            }
            Button {
                Task { await viewModel.downloadModel() }
            } label: {
                Text("Download Model")
                    .font(.headline)
                    .padding(.horizontal, 32)
                    .padding(.vertical, 12)
            }
            .buttonStyle(.borderedProminent)
            .tint(.blue)
            Spacer()
        }
        .padding()
    }

    private func downloadProgressView(progress: Double) -> some View {
        VStack(spacing: 20) {
            Spacer()
            ZStack {
                Circle().stroke(Color.blue.opacity(0.15), lineWidth: 6).frame(width: 80, height: 80)
                Circle().trim(from: 0, to: progress)
                    .stroke(Color.blue, style: StrokeStyle(lineWidth: 6, lineCap: .round))
                    .frame(width: 80, height: 80)
                    .rotationEffect(.degrees(-90))
                    .animation(.easeInOut(duration: 0.3), value: progress)
                Text("\(Int(progress * 100))%")
                    .font(.title3.weight(.semibold).monospacedDigit())
            }
            VStack(spacing: 4) {
                Text("Downloading \(viewModel.selectedModel.displayName)")
                    .font(.headline)
                Text("This may take a few minutes on first launch")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private func errorView(message: String) -> some View {
        VStack(spacing: 20) {
            Spacer()
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 44))
                .foregroundStyle(.orange)
            VStack(spacing: 6) {
                Text("Something went wrong")
                    .font(.headline)
                Text(message)
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 40)
            }
            Button("Try Again") {
                Task { await viewModel.downloadModel() }
            }
            .buttonStyle(.borderedProminent)
            .tint(.blue)
            Spacer()
        }
        .padding()
    }
}
