import { PersonalAssistant } from './personalAssistant.ts';
import * as readline from 'readline';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

async function main() {
    const assistant = new PersonalAssistant();
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    console.log("Chat started. Type 'exit' to end the conversation, or 'clear' to reset chat history.");

    const askQuestion = () => {
        rl.question('You: ', async (input) => {
            if (input.toLowerCase() === 'exit') {
                rl.close();
                return;
            }

            if (input.toLowerCase() === 'clear') {
                assistant.clearChatHistory();
                console.log('Chat history cleared.');
                askQuestion();
                return;
            }

            try {
                await assistant.chat(input);
            } catch (error) {
                console.error('Error:', error);
            }

            askQuestion();
        });
    };

    askQuestion();
}

main().catch(console.error);