import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { MemoryManager } from "./memoryManager.ts";

export class Chatbot {
    private model: ChatOpenAI;
    private memory: MemoryManager;
    private activeThreads: Map<string, string[]>;  // Track conversation history per thread

    constructor() {
        this.memory = new MemoryManager();
        this.model = new ChatOpenAI({
            modelName: "gpt-4o-mini",
            temperature: 0.7
        });
        this.activeThreads = new Map();
    }

    /**
     * Main chat function with Pinecone context retrieval.
     * The incoming message is enhanced with context from Pinecone if available.
     *
     * @param message User input.
     * @param threadId ID for the conversation thread.
     * @returns Assistant's response.
     */
    async chat(message: string, threadId: string = "default"): Promise<string> {
        // Retrieve relevant memories using getRelatedMemories
        const relatedMemories = await this.memory.getRelatedMemories(message);
        let context = "";
        if (relatedMemories.length > 0) {
            context = relatedMemories.map((mem: { metadata: { summary: string } }) => mem.metadata.summary).join("\n");
        }

        const systemPrompt = context
            ? `You are a helpful assistant. Here is some background context: ${context}`
            : "You are a helpful assistant.";

        const messages = [
            new SystemMessage(systemPrompt),
            new HumanMessage(message),
        ];

        // Invoke the language model.
        const response = await this.model.invoke(messages);
        const content = response.content;
        const finalContent =
            typeof content === "string"
                ? content
                : Array.isArray(content)
                    ? content.join(" ")
                    : "";
        const trimmedResponse = finalContent.trim();

        // Store conversation parts in Pinecone for later retrieval.
        await this.memory.addMemory(message, "conversation", [], []);

        // Still update thread history if needed.
        this.updateThreadHistory(threadId, message);
        
        return trimmedResponse;
    }

    private updateThreadHistory(threadId: string, conversation: string) {
        const history = this.activeThreads.get(threadId) || [];
        history.push(conversation);
        if (history.length > 10) history.shift();
        this.activeThreads.set(threadId, history);
    }

    async endThread(threadId: string) {
        this.activeThreads.delete(threadId);
    }

    async *chatStream(message: string, threadId: string = "default"): AsyncIterable<{ content: string }> {
        const response = await this.chat(message, threadId);
        const chunkSize = 1;
        for (let i = 0; i < response.length; i += chunkSize) {
            await new Promise(resolve => setTimeout(resolve, 20));
            yield { content: response.substring(i, i + chunkSize) };
        }
    }
}