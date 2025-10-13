"use client";

import React, { useState, useRef, useEffect } from "react";
import ChatMessages, { ChatMessageType } from "./ChatMessages";
import TextArea from "./TextArea";

const Rag: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const handleSend = async () => {
  if (!text.trim()) return;

  const userMessage = text.trim();
  setText("");

  // Add the user's message immediately, with bot "..." placeholder
  const newMessages = [
    ...messages,
    { userRequest: userMessage, botResponse: "..." },
  ];
  setMessages(newMessages);
  setLoading(true);

  try {
    const res = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: userMessage }),
    });

    const data = await res.json();

    // Replace "..." with backend response
    setMessages((prev) =>
      prev.map((m, i) =>
        i === prev.length - 1 ? { ...m, botResponse: data.response } : m
      )
    );
  } catch (err) {
    console.error("Error contacting backend:", err);
    setMessages((prev) =>
      prev.map((m, i) =>
        i === prev.length - 1
          ? { ...m, botResponse: "❌ Error connecting to backend." }
          : m
      )
    );
  } finally {
    setLoading(false);
  }
};

  // Auto-scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex flex-col h-[90vh] bg-gray-50 rounded-xl overflow-hidden">
      {/* Chat messages */}
      <div className="flex-1 overflow-y-auto px-3 sm:px-5 py-4">
        <ChatMessages messages={messages} />
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="sticky bottom-0 w-full bg-white border-t">
        <div className="max-w-2xl mx-auto px-2 sm:px-4">
          <TextArea
            text={text}
            setText={setText}
            onSend={handleSend}
            disabled={loading}
          />
        </div>
      </div>
    </div>
  );
};

export default Rag;
