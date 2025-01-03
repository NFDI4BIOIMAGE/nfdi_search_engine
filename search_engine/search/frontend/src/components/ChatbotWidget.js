import React, { useState } from 'react';
import "../assets/styles/style.css";
import robotIcon from "../assets/images/robot.png";
import axios from "axios";

const ChatbotWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");

  const toggleChatbot = () => {
    setIsOpen(!isOpen);
  };

  const handleQueryChange = (e) => {
    setQuery(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post("http://localhost:5002/api/chat", { query });
      setResponse(res.data.response);
    } catch (error) {
      setResponse("Error: Unable to get a response from the chatbot.");
    }
  };

  // The return block here defines the UI structure
  return (
    <div className="chatbot-icon-container">
      <img
        src={robotIcon}
        alt="Chatbot Icon"
        className="chatbot-icon"
        onClick={toggleChatbot}
      />
      {isOpen && (
        <div className="chatbot-popup">
          <div className="chatbot-header">
            <h5>NFDIBIOIMAGE Assistant</h5>
            <button onClick={toggleChatbot}>&times;</button>
          </div>
          <div className="chatbot-body">
            <form onSubmit={handleSubmit}>
              <textarea
                value={query}
                onChange={handleQueryChange}
                placeholder="Ask me anything..."
              ></textarea>
              <button type="submit">Send</button>
            </form>
            <div className="chatbot-response">
              <p>{response}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatbotWidget;
