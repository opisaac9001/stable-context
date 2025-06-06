// yawl/gui/src/components/ChatView.jsx
import React, { useState, useEffect, useRef } from 'react';
import { sendMessage } from '../apiClient'; // Assuming apiClient.js is in ../

function ChatView() {
    const [messages, setMessages] = useState([]);
    const [currentInput, setCurrentInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null); // For auto-scrolling

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleInputChange = (event) => {
        setCurrentInput(event.target.value);
    };

    const handleSendMessage = async () => {
        if (!currentInput.trim() || isLoading) return;

        const userMessage = { role: 'user', content: currentInput.trim() };
        setMessages(prevMessages => [...prevMessages, userMessage]);
        setIsLoading(true);
        const messageToSend = currentInput.trim();
        setCurrentInput(''); // Clear input after sending

        try {
            // Default useRag: false, default generationParams: {}
            const response = await sendMessage(messageToSend);

            if (response.success && response.data && response.data.reply) {
                const assistantMessage = { role: 'assistant', content: response.data.reply };
                setMessages(prevMessages => [...prevMessages, assistantMessage]);
            } else {
                const errorMessage = {
                    role: 'system',
                    content: `Error: ${response.error?.detail || response.error?.message || 'Failed to get reply.'}`
                };
                setMessages(prevMessages => [...prevMessages, errorMessage]);
            }
        } catch (error) {
            console.error("Error sending message:", error);
            const errorMessage = {
                role: 'system',
                content: `Error: ${error.message || 'Network error or unexpected issue.'}`
            };
            setMessages(prevMessages => [...prevMessages, errorMessage]);
        }
        setIsLoading(false);
    };

    const handleKeyPress = (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent newline in textarea
            handleSendMessage();
        }
    };

    // Basic inline styles for demonstration - can be moved to CSS file
    const styles = {
        chatContainer: { display: 'flex', flexDirection: 'column', height: '300px', border: '1px solid #ccc', overflowY: 'hidden'},
        messagesArea: { flexGrow: 1, overflowY: 'auto', padding: '10px', display: 'flex', flexDirection: 'column' },
        message: { marginBottom: '5px', padding: '8px', borderRadius: '5px', maxWidth: '80%' },
        userMessage: { backgroundColor: '#dcf8c6', alignSelf: 'flex-end', textAlign: 'right' },
        assistantMessage: { backgroundColor: '#f1f0f0', alignSelf: 'flex-start', textAlign: 'left' },
        systemMessage: { backgroundColor: '#ffebee', color: '#c62828', alignSelf: 'center', fontStyle: 'italic', fontSize: '0.9em' },
        inputArea: { display: 'flex', padding: '10px', borderTop: '1px solid #ccc' },
        textInput: { flexGrow: 1, padding: '8px', border: '1px solid #ddd', borderRadius: '5px', marginRight: '5px', resize: 'none' },
        sendButton: { padding: '8px 15px', border: 'none', backgroundColor: '#007bff', color: 'white', borderRadius: '5px', cursor: 'pointer' },
        disabledButton: { backgroundColor: '#ccc' }
    };

    return (
        <div style={styles.chatContainer}>
            <div style={styles.messagesArea}>
                {messages.map((msg, index) => (
                    <div
                        key={index}
                        style={{
                            ...styles.message,
                            ...(msg.role === 'user' ? styles.userMessage :
                                msg.role === 'assistant' ? styles.assistantMessage :
                                styles.systemMessage)
                        }}
                    >
                        {msg.content}
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>
            <div style={styles.inputArea}>
                <textarea
                    value={currentInput}
                    onChange={handleInputChange}
                    onKeyPress={handleKeyPress}
                    style={styles.textInput}
                    placeholder="Type your message..."
                    rows={2}
                    disabled={isLoading}
                />
                <button
                    onClick={handleSendMessage}
                    disabled={isLoading}
                    style={{...styles.sendButton, ...(isLoading ? styles.disabledButton : {})}}
                >
                    {isLoading ? 'Sending...' : 'Send'}
                </button>
            </div>
        </div>
    );
}

export default ChatView;
