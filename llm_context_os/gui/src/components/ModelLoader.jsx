// llm_context_os/gui/src/components/ModelLoader.jsx
import React, { useState } from 'react';
import { loadModel } from '../apiClient';

const MODEL_TYPES = ['api', 'gguf', 'awq', 'exl2']; // Add other types as needed

function ModelLoader() {
    const [modelPathOrName, setModelPathOrName] = useState('');
    const [modelType, setModelType] = useState(MODEL_TYPES[0]);
    const [runnerParamsStr, setRunnerParamsStr] = useState('{}'); // Store as string
    const [idleUnloadSecStr, setIdleUnloadSecStr] = useState('900'); // Store as string
    const [statusMessage, setStatusMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleLoadModel = async () => {
        if (!modelPathOrName.trim() || !modelType) {
            setStatusMessage('Error: Model Path/Name and Model Type are required.');
            return;
        }
        setIsLoading(true);
        setStatusMessage('Loading model...');

        let parsedRunnerParams = {};
        try {
            // Ensure empty string for runnerParamsStr defaults to empty object
            parsedRunnerParams = JSON.parse(runnerParamsStr.trim() === '' ? '{}' : runnerParamsStr);
        } catch (e) {
            setStatusMessage(`Error: Invalid JSON in Runner Params: ${e.message}`);
            setIsLoading(false);
            return;
        }

        let parsedIdleUnloadSec = null; // Default to null (no change to manager's default)
        if (idleUnloadSecStr.trim() !== '') {
            const num = parseInt(idleUnloadSecStr, 10);
            if (isNaN(num) || num < 0) { // 0 is allowed (disable idle unload)
                setStatusMessage('Error: Idle Unload Seconds must be a non-negative number.');
                setIsLoading(false);
                return;
            }
            parsedIdleUnloadSec = num;
        }

        try {
            const response = await loadModel(modelType, modelPathOrName.trim(), parsedRunnerParams, parsedIdleUnloadSec);
            if (response.success && response.data) {
                setStatusMessage(`Status: ${response.data.status || 'unknown'} - ${response.data.message || 'Request processed.'}`);
            } else {
                setStatusMessage(`Error: ${response.error?.detail || response.error?.message || 'Failed to load model.'}`);
            }
        } catch (error) {
            console.error("Error loading model:", error);
            setStatusMessage(`Error: ${error.message || 'Network error or unexpected issue.'}`);
        }
        setIsLoading(false);
    };

    // Basic inline styles for demonstration
    const styles = {
        loaderContainer: { padding: '15px', border: '1px solid #eee', borderRadius: '5px', marginBottom: '20px' },
        inputGroup: { marginBottom: '10px' },
        label: { display: 'block', marginBottom: '3px', fontWeight: 'bold', fontSize: '0.9em' },
        input: { width: 'calc(100% - 16px)', padding: '8px', border: '1px solid #ccc', borderRadius: '3px', boxSizing: 'border-box' },
        select: { width: '100%', padding: '8px', border: '1px solid #ccc', borderRadius: '3px', boxSizing: 'border-box' },
        textarea: { width: 'calc(100% - 16px)', padding: '8px', border: '1px solid #ccc', borderRadius: '3px', minHeight: '40px', boxSizing: 'border-box', fontFamily: 'monospace' },
        button: { padding: '10px 15px', border: 'none', backgroundColor: '#28a745', color: 'white', borderRadius: '5px', cursor: 'pointer', fontSize: '1em' },
        disabledButton: { backgroundColor: '#ccc' },
        status: { marginTop: '10px', padding: '10px', border: '1px dashed #ccc', borderRadius: '3px', background: '#f8f9fa' }
    };

    return (
        <div style={styles.loaderContainer}>
            <h2>Load Model</h2>
            <div style={styles.inputGroup}>
                <label htmlFor="modelPath" style={styles.label}>Model Path/Name/Repo ID:</label>
                <input type="text" id="modelPath" style={styles.input} value={modelPathOrName} onChange={e => setModelPathOrName(e.target.value)} disabled={isLoading} />
            </div>
            <div style={styles.inputGroup}>
                <label htmlFor="modelType" style={styles.label}>Model Type:</label>
                <select id="modelType" style={styles.select} value={modelType} onChange={e => setModelType(e.target.value)} disabled={isLoading}>
                    {MODEL_TYPES.map(type => <option key={type} value={type}>{type.toUpperCase()}</option>)}
                </select>
            </div>
            <div style={styles.inputGroup}>
                <label htmlFor="runnerParams" style={styles.label}>Runner Params (JSON):</label>
                <textarea id="runnerParams" style={styles.textarea} value={runnerParamsStr} onChange={e => setRunnerParamsStr(e.target.value)} disabled={isLoading} placeholder='e.g., {"api_url": "http://...", "n_gpu_layers": 20}' />
            </div>
            <div style={styles.inputGroup}>
                <label htmlFor="idleUnload" style={styles.label}>Idle Unload Sec (0=disable, blank=manager default):</label>
                <input type="text" id="idleUnload" style={styles.input} value={idleUnloadSecStr} onChange={e => setIdleUnloadSecStr(e.target.value)} disabled={isLoading} placeholder='e.g., 900' />
            </div>
            <button onClick={handleLoadModel} disabled={isLoading} style={{...styles.button, ...(isLoading ? styles.disabledButton : {})}}>
                {isLoading ? 'Loading...' : 'Load Model'}
            </button>
            {statusMessage && <div style={styles.status}><p>{statusMessage}</p></div>}
        </div>
    );
}

export default ModelLoader;
