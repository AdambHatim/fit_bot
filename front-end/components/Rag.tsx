"use client";

import React, { useState, useRef, useEffect } from "react";
import ChatMessages, { ChatMessageType } from "./ChatMessages";
import TextArea from "./TextArea";

const Rag: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const handleSend = () => {
    if (!text.trim()) return;

    const userMessage = text.trim();
    setText("");

    const newMessages = [
      ...messages,
      { userRequest: userMessage, botResponse: "..." },
    ];
    setMessages(newMessages);
    setLoading(true);

    setTimeout(() => {
      setLoading(false);
    }, 1000);
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
