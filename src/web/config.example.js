/**
 * MRI-Crohn Atlas Configuration
 *
 * SETUP INSTRUCTIONS:
 * 1. Copy this file to config.js in the same directory
 * 2. Replace the placeholder with your actual OpenRouter API key
 * 3. config.js is gitignored - never commit your actual API key
 *
 * Get an API key at: https://openrouter.ai/keys
 */

const CONFIG = {
    OPENROUTER_API_KEY: 'your-openrouter-api-key-here'
};

// Do not modify below this line
if (typeof window !== 'undefined') {
    window.CONFIG = CONFIG;
}
