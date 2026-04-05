//
//  GemmaModel.swift
//  gemma4
//

import Foundation

enum GemmaModel: String, CaseIterable, Identifiable {
    case e2b
    case e4b

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .e2b: "Gemma 4 E2B"
        case .e4b: "Gemma 4 E4B"
        }
    }

    var modelId: String {
        switch self {
        case .e2b: "mlx-community/gemma-4-E2B-it-4bit"
        case .e4b: "mlx-community/gemma-4-e4b-it-4bit"
        }
    }

    var description: String {
        switch self {
        case .e2b: "2B params \u{2022} Fastest"
        case .e4b: "4B params \u{2022} Smarter"
        }
    }

}
