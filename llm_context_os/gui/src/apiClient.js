// yawl/gui/src/apiClient.js
const API_BASE_URL = 'http://localhost:8000'; // Assuming backend runs on port 8000

async function request(endpoint, method = 'GET', body = null) {
    const url = `${API_BASE_URL}${endpoint}`;
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
        },
    };
    if (body) {
        options.body = JSON.stringify(body);
    }

    try {
        const response = await fetch(url, options);
        if (!response.ok) {
            // Attempt to parse error response from API if available
            let errorData;
            try {
                errorData = await response.json();
            } catch (e) {
                errorData = { message: `HTTP error ${response.status}` };
            }
            console.error('API request failed:', response.status, errorData);
            return { success: false, error: errorData, status: response.status };
        }
        // Check if response is JSON before parsing
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
             const data = await response.json();
             return { success: true, data: data, status: response.status };
        }
        return { success: true, data: {}, status: response.status }; // No JSON content

    } catch (error) {
        console.error('Network or other error:', error);
        return { success: false, error: { message: error.message }, status: 0 };
    }
}

export async function loadModel(modelType, modelPathOrName, runnerParams = {}, idleUnloadSec = null) {
    const payload = {
        model_type: modelType,
        model_path_or_name: modelPathOrName,
        runner_params: { ...runnerParams }, // Ensure runner_params is always an object
    };
    // Add idle_unload_sec to runner_params if it's explicitly provided and valid
    if (idleUnloadSec !== null && typeof idleUnloadSec === 'number') {
      payload.runner_params.idle_unload_sec = idleUnloadSec;
    }
    return request('/load_model', 'POST', payload);
}

export async function sendMessage(message, useRag = false, generationParams = {}) {
    const payload = {
        message: message,
        use_rag: useRag,
        generation_params: generationParams,
    };
    return request('/chat', 'POST', payload);
}

// Optional: Add a simple test function here for development, e.g.:
async function testApiClient() {
    console.log('Testing apiClient...');
    // Note: Requires backend server to be running.

    // Test loadModel (using a placeholder type that should fail gracefully or succeed if manager handles it)
    // For this test to show a "success" from the API, the backend's ModelManager.load()
    // would need to handle this 'api' type correctly and the placeholder APIRunner would init.
    // The important part for apiClient is that the request is structured correctly.
    console.log("Attempting to load a dummy 'api' model...");
    const loadRes = await loadModel(
        'api', // modelType
        'test-model-from-gui', // modelPathOrName (model_name for APIRunner)
        { api_url: 'http://dummyhost:1234/v1' }, // runnerParams
        300 // idleUnloadSec
    );
    console.log('Load model response:', loadRes);

    if (loadRes.success && loadRes.data && loadRes.data.status === 'ok') {
       console.log('Model loaded successfully, attempting chat...');
       const chatRes = await sendMessage('Hello from apiClient test!', false, { max_new_tokens: 50 });
       console.log('Chat response:', chatRes);
    } else {
       console.log('Model not loaded or load failed. Testing chat error handling (expect 400 if no model loaded)...');
       const chatResFail = await sendMessage('Hello when no model loaded');
       console.log('Chat response (expected error if no model):', chatResFail);
    }

    console.log("\nTesting load with an invalid model type (from GUI perspective):");
    const invalidLoadRes = await loadModel('unknown_gui_type', 'some-model-path');
    console.log('Invalid load model response:', invalidLoadRes);
}

// To run the test: open browser console on a page where this script is loaded and type testApiClient()
// or include a call here if not expecting this to be a module strictly imported.
// For now, it's better to keep it as a function to be called manually during dev.
// Example: if (typeof window !== 'undefined') { window.testApiClient = testApiClient; }
// testApiClient(); // Do not uncomment for actual build, as it requires a running backend.
