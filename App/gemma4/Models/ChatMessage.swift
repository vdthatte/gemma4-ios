//
//  ChatMessage.swift
//  gemma4
//

import Foundation
import SwiftData

// MARK: - ChatMessage

@Model
final class ChatMessage: Identifiable {
    var id: UUID
    var roleRaw: String
    var content: String
    var timestamp: Date
    var conversation: Conversation?
    var sortOrder: Int

    init(role: Role, content: String, sortOrder: Int = 0) {
        self.id = UUID()
        self.roleRaw = role.rawValue
        self.content = content
        self.timestamp = .now
        self.sortOrder = sortOrder
    }

    var role: Role {
        get { Role(rawValue: roleRaw) ?? .user }
        set { roleRaw = newValue.rawValue }
    }

    enum Role: String, Codable {
        case user
        case assistant
        case system
    }
}

extension ChatMessage {
    static func user(_ content: String, sortOrder: Int = 0) -> ChatMessage {
        ChatMessage(role: .user, content: content, sortOrder: sortOrder)
    }

    static func assistant(_ content: String, sortOrder: Int = 0) -> ChatMessage {
        ChatMessage(role: .assistant, content: content, sortOrder: sortOrder)
    }

    static func system(_ content: String, sortOrder: Int = 0) -> ChatMessage {
        ChatMessage(role: .system, content: content, sortOrder: sortOrder)
    }
}

// MARK: - Conversation

@Model
final class Conversation: Identifiable {
    var id: UUID
    var title: String
    var createdAt: Date

    @Relationship(deleteRule: .cascade, inverse: \ChatMessage.conversation)
    var messages: [ChatMessage]

    init(title: String = "New Chat") {
        self.id = UUID()
        self.title = title
        self.createdAt = .now
        self.messages = []
    }

    var sortedMessages: [ChatMessage] {
        messages.sorted { $0.sortOrder < $1.sortOrder }
    }

    var displayTitle: String {
        if let firstUserMessage = sortedMessages.first(where: { $0.role == .user }) {
            let text = firstUserMessage.content
            return String(text.prefix(40)) + (text.count > 40 ? "..." : "")
        }
        return title
    }
}
