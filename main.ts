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

    console.log("Chat started. Type 'exit' to end the conversation.");

    const askQuestion = () => {
        rl.question('You: ', async (input) => {
            if (input.toLowerCase() === 'exit') {
                rl.close();
                return;
            }

            try {
                process.stdout.write('Assistant: ');
                const response = await assistant.chat(input);
                process.stdout.write(response);
                console.log('\n'); // New line after response
            } catch (error) {
                console.error('Error:', error);
            }

            askQuestion();
        });
    };

    askQuestion();
}

main().catch(console.error);