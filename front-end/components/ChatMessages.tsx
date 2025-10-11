"use client";

import React from "react";

export interface ChatMessageType {
  userRequest: string;
  botResponse: string;
}

interface ChatMessagesProps {
  messages: ChatMessageType[];
}

const ChatMessages: React.FC<ChatMessagesProps> = ({ messages }) => {
  return (
    <div className="flex flex-col space-y-3">
      {messages.map((msg, index) => (
        <div key={index} className="flex flex-col space-y-2">
          {/* User */}
          <div className="flex justify-end">
            <div className="max-w-[70%] rounded-xl bg-blue-600 text-white p-2.5 text-sm shadow-md break-words">
              <p>{msg.userRequest}</p>
            </div>
          </div>

          {/* Bot */}
          <div className="flex justify-start">
            <div className="max-w-[70%] rounded-xl bg-gray-200 text-gray-800 p-2.5 text-sm shadow-sm break-words">
              <p>{msg.botResponse}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default ChatMessages;
