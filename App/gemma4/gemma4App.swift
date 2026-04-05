//
//  gemma4App.swift
//  gemma4
//

import SwiftUI
import SwiftData

@main
struct gemma4App: App {
    @State private var mlxService = MLXService()
    @State private var viewModel: ChatViewModel?

    let modelContainer: ModelContainer = {
        let schema = Schema([
            Conversation.self,
            ChatMessage.self,
        ])
        let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)
        do {
            return try ModelContainer(for: schema, configurations: [config])
        } catch {
            fatalError("Could not create ModelContainer: \(error)")
        }
    }()

    var body: some Scene {
        WindowGroup {
            if let viewModel {
                ContentView(viewModel: viewModel)
            } else {
                Color.clear.onAppear {
                    let context = modelContainer.mainContext
                    viewModel = ChatViewModel(mlxService: mlxService, modelContext: context)
                }
            }
        }
        .modelContainer(modelContainer)
    }
}
