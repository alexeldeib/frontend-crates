// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Based on https://github.com/64bit/async-openai/ by Himanshu Neema
// Original Copyright (c) 2022 Himanshu Neema
// Licensed under MIT License (see ATTRIBUTIONS-Rust.md)
//
// Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
// Licensed under Apache 2.0

use dynamo_protocols::types::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    ChatCompletionResponseMessage, ChatCompletionStreamResponseDelta, CreateChatCompletionRequest,
    CreateChatCompletionRequestArgs, ReasoningContent,
};

#[tokio::test]
async fn chat_types_serde() {
    let request: CreateChatCompletionRequest = CreateChatCompletionRequestArgs::default()
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content("your are a calculator")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("what is the result of 1+1")
                .build()
                .unwrap()
                .into(),
        ])
        .build()
        .unwrap();
    // serialize the request
    let serialized = serde_json::to_string(&request).unwrap();
    // deserialize the request
    let deserialized: CreateChatCompletionRequest = serde_json::from_str(&serialized).unwrap();
    assert_eq!(request, deserialized);
}

#[test]
fn assistant_reasoning_alias_normalizes_to_canonical_field() {
    let cases = [
        (
            serde_json::json!("thinking"),
            ReasoningContent::Text("thinking".to_string()),
        ),
        (
            serde_json::json!(["before", "after"]),
            ReasoningContent::Segments(vec!["before".to_string(), "after".to_string()]),
        ),
    ];

    for (value, expected) in cases {
        let canonical: ChatCompletionRequestAssistantMessage =
            serde_json::from_value(serde_json::json!({"reasoning_content": value.clone()}))
                .unwrap();
        let alias: ChatCompletionRequestAssistantMessage =
            serde_json::from_value(serde_json::json!({"reasoning": value.clone()})).unwrap();

        assert_eq!(canonical, alias);
        assert_eq!(alias.reasoning_content, Some(expected));
        assert_eq!(
            serde_json::to_value(alias).unwrap(),
            serde_json::json!({"reasoning_content": value})
        );
    }
}

#[test]
fn chat_request_accepts_reasoning_alias_for_assistant_message() {
    let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "messages": [{
            "role": "assistant",
            "content": null,
            "reasoning": "thinking"
        }],
        "model": "test-model"
    }))
    .unwrap();

    let [ChatCompletionRequestMessage::Assistant(message)] = request.messages.as_slice() else {
        panic!("expected one assistant message");
    };
    assert_eq!(
        message.reasoning_content,
        Some(ReasoningContent::Text("thinking".to_string()))
    );
}

#[test]
fn response_reasoning_alias_round_trips_with_canonical_field() {
    let canonical: ChatCompletionResponseMessage = serde_json::from_value(serde_json::json!({
        "content": null,
        "role": "assistant",
        "reasoning_content": "thinking"
    }))
    .unwrap();
    let alias: ChatCompletionResponseMessage = serde_json::from_value(serde_json::json!({
        "content": null,
        "role": "assistant",
        "reasoning": "thinking"
    }))
    .unwrap();

    assert_eq!(canonical, alias);
    let serialized = serde_json::to_value(&alias).unwrap();
    assert_eq!(
        serialized,
        serde_json::json!({
            "content": null,
            "role": "assistant",
            "reasoning_content": "thinking"
        })
    );
    assert_eq!(
        serde_json::from_value::<ChatCompletionResponseMessage>(serialized).unwrap(),
        alias
    );
}

#[test]
fn stream_reasoning_alias_round_trips_with_canonical_field() {
    let canonical: ChatCompletionStreamResponseDelta =
        serde_json::from_value(serde_json::json!({"reasoning_content": "thinking"})).unwrap();
    let alias: ChatCompletionStreamResponseDelta =
        serde_json::from_value(serde_json::json!({"reasoning": "thinking"})).unwrap();

    assert_eq!(canonical, alias);
    let serialized = serde_json::to_value(&alias).unwrap();
    assert_eq!(
        serialized,
        serde_json::json!({"reasoning_content": "thinking"})
    );
    assert_eq!(
        serde_json::from_value::<ChatCompletionStreamResponseDelta>(serialized).unwrap(),
        alias
    );
}

#[test]
fn reasoning_alias_rejects_ambiguous_input() {
    fn assert_duplicate_field<T>(json: &str)
    where
        T: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let error = serde_json::from_str::<T>(json).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("duplicate field `reasoning_content`"),
            "unexpected error: {error}"
        );
    }

    assert_duplicate_field::<ChatCompletionRequestAssistantMessage>(
        r#"{"reasoning_content":"canonical","reasoning":"alias"}"#,
    );
    assert_duplicate_field::<ChatCompletionResponseMessage>(
        r#"{"content":null,"role":"assistant","reasoning_content":"canonical","reasoning":"alias"}"#,
    );
    assert_duplicate_field::<ChatCompletionStreamResponseDelta>(
        r#"{"reasoning_content":"canonical","reasoning":"alias"}"#,
    );
}
