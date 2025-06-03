// llm_context_os/gui/src/App.jsx
import React from 'react';
import ModelLoader from './components/ModelLoader';
import ChatView from './components/ChatView';

function App() {
  // Basic inline styles for layout
  const styles = {
    appContainer: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '20px',
      fontFamily: 'sans-serif'
    },
    header: {
      marginBottom: '20px',
      textAlign: 'center'
    },
    contentContainer: {
      display: 'flex',
      flexDirection: 'column',
      width: '100%',
      maxWidth: '800px', // Limit overall width
      gap: '20px' // Space between ModelLoader and ChatView
    }
  };

  return (
    <div style={styles.appContainer}>
      <header style={styles.header}>
        <h1>LLM Context OS - GUI</h1>
      </header>
      <div style={styles.contentContainer}>
        <ModelLoader />
        <ChatView />
      </div>
      {/* Future: Could add SettingsView, ToolView etc. here or manage via routing */}
    </div>
  );
}

export default App;
