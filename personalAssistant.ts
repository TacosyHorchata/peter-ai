import { OpenAI } from 'openai';
import { MemoryManager } from './memoryManager.ts';

export class PersonalAssistant {
    private openai: OpenAI;
    private memoryManager: MemoryManager;

    constructor() {
        this.openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
        this.memoryManager = new MemoryManager();
    }

    async chat(userInput: string): Promise<string> {
        try {
            // Get relevant memories
            const relevantMemories = await this.memoryManager.getRelatedMemories(userInput);
            console.log("Retrieved memories:", relevantMemories.length);
            
            // Construct the prompt with context from memories
            const contextPrompt = this.constructPromptWithContext(userInput, relevantMemories);
            
            // Get response from ChatGPT
            const response = await this.openai.chat.completions.create({
                model: "gpt-4o-mini",
                messages: [
                    {
                        role: "system",
                        content: "You are a helpful personal assistant with access to previous conversation memories. Use the provided context when relevant, but don't mention the memory system explicitly to the user."
                    },
                    {
                        role: "user",
                        content: contextPrompt
                    }
                ]
            });
    
            const assistantResponse = response.choices[0].message.content || '';
    
            // Store the complete interaction and let MemoryManager decide if it's worth remembering
            await this.memoryManager.addMemory(
                `User: ${userInput}\nAssistant: ${assistantResponse}`,
                'conversation'
            );
    
            return assistantResponse;
        } catch (error) {
            console.error('Error in chat:', error);
            throw error;
        }
    }
    
    private constructPromptWithContext(userInput: string, relevantMemories: any[]): string {
        let prompt = '';
    
        if (relevantMemories.length > 0) {
            prompt += "Here are relevant facts I know:\n";
            relevantMemories.forEach((memory) => {
                prompt += `- ${memory.metadata.summary}\n`;
            });
            prompt += "\n";
        }
    
        prompt += `Current user input: ${userInput}`;
        return prompt;
    }
}