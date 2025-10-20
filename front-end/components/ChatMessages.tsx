"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export interface ChatMessageType {
  userRequest: string;
  botResponse: string;
}

interface ChatMessagesProps {
  messages: ChatMessageType[];
}

const ChatMessages: React.FC<ChatMessagesProps> = ({ messages }) => {
  return (
    <div className="flex flex-col space-y-4">
      {messages.map((msg, index) => (
        <div key={index} className="flex flex-col space-y-2">
          {/* User message */}
          <div className="flex justify-end">
            <div className="max-w-[70%] rounded-2xl bg-blue-600 text-white p-3 text-sm shadow-md break-words whitespace-pre-wrap">
              {msg.userRequest}
            </div>
          </div>

          {/* Bot message (Markdown) */}
          <div className="flex justify-start">
            <div className="max-w-[75%] rounded-2xl bg-gray-100 text-gray-900 p-3 text-sm shadow-sm break-words prose prose-sm max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.botResponse}
              </ReactMarkdown>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default ChatMessages;
