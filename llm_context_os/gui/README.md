# LLM Context OS - GUI Frontend

This directory contains the source code for the GUI frontend of the LLM Context OS, built using React with Vite, and intended to be packaged with Tauri.

## Prerequisites

- [Node.js](https://nodejs.org/) (which includes npm) - Version 18.x or higher recommended.
- (For Tauri development/build) [Rust and Tauri prerequisites](https://tauri.app/v1/guides/getting-started/prerequisites).

## Development Setup

1.  **Navigate to the GUI directory:**
    ```bash
    cd llm_context_os/gui
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

## Running the Frontend (Vite Dev Server)

This will run the React application in a web browser, typically for development and testing the UI components against the backend API (ensure the Python backend is running separately).

```bash
npm run dev
```
The application should then be accessible at `http://localhost:5173` (or another port if 5173 is busy).

## Building the Frontend for Tauri

This command compiles the React app into static assets that Tauri will use.

```bash
npm run build
```
The output will be in the `llm_context_os/gui/dist` directory, which is referenced by `tauri.conf.json`.

## Tauri Integration

The `src-tauri` directory contains the Tauri-specific configuration (`tauri.conf.json`) and Rust code (if any, not yet implemented significantly).

To run the application as a Tauri desktop app (after installing Tauri CLI and prerequisites):

1.  Ensure the backend Python server is running.
2.  Navigate to the Tauri source directory:
    ```bash
    cd llm_context_os/gui/src-tauri
    ```
3.  Run the Tauri development command:
    ```bash
    cargo tauri dev
    ```

To build the Tauri application into an executable:
```bash
cd llm_context_os/gui/src-tauri
cargo tauri build
```
This project provides a functional skeleton. Styling and advanced UI features are to be implemented.
