import OpenAI from 'openai';
type ChatCompletionMessageParam = OpenAI.ChatCompletionMessageParam;
import { MemoryManager } from './memoryManager.ts';

export class PersonalAssistant {
    private openai: OpenAI;
    private memoryManager: MemoryManager;
    private currentChatHistory: { role: string; content: string }[] = [];

    constructor() {
        this.openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
        this.memoryManager = new MemoryManager();
    }

    async chat(userInput: string): Promise<string> {
        try {
            // Add user input to chat history
            this.currentChatHistory.push({ role: "user", content: userInput });
            
            // Keep only last 5 messages
            if (this.currentChatHistory.length > 5) {
                this.currentChatHistory = this.currentChatHistory.slice(-5);
            }
            
            // Get relevant memories with a lower similarity threshold
            const relevantMemories = await this.memoryManager.getRelatedMemories(userInput, 5); // Add parameters for count and threshold
            console.log("Retrieved memories:", relevantMemories.length);
            
            // Construct messages array with history and context
            const messages = [
                {
                    role: "system",
                    content: "You are Peter, a helpful personal assistant with access to previous conversation memories. Use the provided context when relevant."
                },
                // Add context from memories
                {
                    role: "system",
                    content: this.constructContextFromMemories(relevantMemories)
                },
                // Add chat history
                ...this.currentChatHistory
            ];

            // Use correct model name
            const response = await this.openai.chat.completions.create({
                model: "gpt-4o-mini",
                messages: messages as ChatCompletionMessageParam[],
                temperature: 0.7
            });

            const fullResponse = response.choices[0].message.content || '';
            console.log('Assistant:', fullResponse);

            // Add assistant's response to chat history
            this.currentChatHistory.push({ role: "assistant", content: fullResponse });
            
            // Keep chat history manageable (last 10 messages)
            if (this.currentChatHistory.length > 10) {
                this.currentChatHistory = this.currentChatHistory.slice(-10);
            }

            // Store the complete interaction
            await this.memoryManager.addMemory(
                `User: ${userInput}\nAssistant: ${fullResponse}`,
                'conversation'
            );

            return fullResponse;
        } catch (error) {
            console.error('Error in chat:', error);
            throw error;
        }
    }
    
    private constructContextFromMemories(relevantMemories: any[]): string {
        if (relevantMemories.length === 0) return '';
        
        return "Here are relevant facts I know:\n" + 
            relevantMemories.map(memory => `- ${memory.metadata.summary}`).join('\n');
    }

    // Add method to clear chat history
    clearChatHistory(): void {
        this.currentChatHistory = [];
    }
}