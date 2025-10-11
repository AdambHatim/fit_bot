"use client";

import React from "react";
import { Send } from "lucide-react";

interface TextAreaProps {
  text: string;
  setText: (val: string) => void;
  onSend: () => void;
  disabled?: boolean;
}

const TextArea: React.FC<TextAreaProps> = ({ text, setText, onSend, disabled }) => {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && text.trim()) onSend();
    }
  };

  return (
    <div className="flex items-center space-x-2 py-2 bg-white">
      <textarea
        className="flex-1 resize-none border rounded-lg p-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        placeholder="Type your message..."
        rows={1}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
      />
      <button
        onClick={onSend}
        disabled={!text.trim() || disabled}
        className="p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg disabled:opacity-50 flex items-center justify-center"
      >
        <Send size={16} />
      </button>
    </div>
  );
};

export default TextArea;
